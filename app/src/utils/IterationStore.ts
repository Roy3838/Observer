// src/utils/IterationStore.ts
//
// Each iteration is written to IndexedDB once, when it finishes (iteration-end / iteration-skipped).
// Memory holds only what the UI polls synchronously: the running sessions' iterations, with the
// images of each agent's last KEEP_IMAGES finished iterations plus any in flight. Everything else
// is read back from IndexedDB.
//
//   observer-iterations v1
//     iterations  key: id   index agent_time: [agentId, startTime]   text only, no base64
//     images      key: id                                            { id, images: string[] }

import { LogEntry, Logger } from './logging';

export interface SensorData {
  type: 'screenshot' | 'camera' | 'ocr' | 'audio' | 'clipboard' | 'memory' | 'imemory';
  content: any;
  timestamp: string;
  size?: number; // For images
  source?: string; // For audio (microphone, screenAudio, allAudio) or imemory (sourceAgent)
  imageCount?: number; // For imemory sensor
}

export interface ToolCall {
  name: string;
  status: 'success' | 'error';
  params?: any;
  error?: string;
  timestamp: string;
}

export interface IterationData {
  id: string;
  agentId: string;
  sessionId: string;
  sessionIterationNumber: number; // 1, 2, 3... within this session
  startTime: string;
  endTime?: string; // Set when it finishes: every tool call is in
  sensors: SensorData[];
  modelPrompt?: string;
  modelImages?: string[]; // Base64 images sent to model; absent on text-only reads (see getRuns({ images: true }))
  modelImageCount?: number;
  modelResponse?: string;
  modelResponseTime?: string;
  tools: ToolCall[];
  duration?: number; // Seconds, start to finish
  bytes?: number; // Approximate stored size
  hasError: boolean;
  isSkipped?: boolean; // True if iteration was skipped
  skipReason?: 'same_inputs' | 'network_error'; // Why the iteration was skipped
}

export interface AgentSession {
  sessionId: string;
  agentId: string;
  startTime: string;
  endTime?: string;
  iterations: IterationData[];
}

// ---- IndexedDB, in three helpers ---------------------------------------------

const req = <T>(r: IDBRequest<T>) =>
  new Promise<T>((ok, fail) => { r.onsuccess = () => ok(r.result); r.onerror = () => fail(r.error); });

const done = (tx: IDBTransaction) =>
  new Promise<void>((ok, fail) => { tx.oncomplete = () => ok(); tx.onerror = tx.onabort = () => fail(tx.error); });

let dbPromise: Promise<IDBDatabase> | null = null;
const db = () => dbPromise ??= new Promise<IDBDatabase>((ok, fail) => {
  const open = indexedDB.open('observer-iterations', 1);
  open.onupgradeneeded = () => {
    open.result.createObjectStore('iterations', { keyPath: 'id' }).createIndex('agent_time', ['agentId', 'startTime']);
    open.result.createObjectStore('images', { keyPath: 'id' });
  };
  open.onsuccess = () => ok(open.result);
  open.onerror = () => fail(open.error);
}).catch(e => { dbPromise = null; throw e; });

// ISO timestamps sort as strings, so a time window is a key range on the index.
const agentRange = (agentId: string, since = '', until = '￿') =>
  IDBKeyRange.bound([agentId, since], [agentId, until]);

// Finished iterations per agent that keep their images in memory (the log viewer shows 50).
const KEEP_IMAGES = 50;

const SENSORS = new Set(['screenshot', 'camera', 'ocr', 'audio', 'clipboard', 'memory', 'imemory']);
const textOf = ({ modelImages, ...text }: IterationData): IterationData => text;
const newestFirst = (a: IterationData, b: IterationData) => b.startTime.localeCompare(a.startTime);

// ---- Store -------------------------------------------------------------------

class IterationStoreClass {
  private live = new Map<string, IterationData>();                     // running sessions + in-flight
  private sessions = new Map<string, { id: string; count: number }>(); // agentId → current session
  private withImages = new Map<string, string[]>();                     // agentId → finished iterations still holding images
  private writes: Promise<void> = Promise.resolve();                    // serialized; every read awaits it

  constructor() {
    Logger.addListener(log => this.onLog(log));
  }

  private onLog(log: LogEntry) {
    const id = log.details?.iterationId;
    const type: string | undefined = log.details?.logType;
    if (!id || !type) return;
    const c = log.details.content;
    const at = log.timestamp.toISOString();

    if (type === 'iteration-start') {
      const session = this.sessions.get(log.source);
      if (session) this.live.set(id, {
        id, agentId: log.source, sessionId: session.id, sessionIterationNumber: ++session.count,
        startTime: at, sensors: [], tools: [], hasError: false,
      });
      return;
    }

    const it = this.live.get(id);
    if (!it) return;

    if (type.startsWith('sensor-') && SENSORS.has(type.slice(7))) {
      it.sensors.push({
        type: type.slice(7) as SensorData['type'], content: c, timestamp: at,
        size: c?.size, source: c?.source ?? c?.sourceAgent, imageCount: c?.imageCount,
      });
    } else if (type === 'model-prompt') {
      it.modelPrompt = typeof c === 'object' ? c.modifiedPrompt : c;
      it.modelImages = typeof c === 'object' ? c.images ?? [] : [];
    } else if (type === 'model-response') {
      it.modelResponse = typeof c === 'string' ? c : '';
      it.modelResponseTime = at;
    } else if (type === 'tool-success' || type === 'tool-error') {
      it.tools.push({
        name: c?.tool ?? 'unknown', status: type === 'tool-success' ? 'success' : 'error',
        params: c?.params, error: c?.error, timestamp: at,
      });
      it.hasError ||= type === 'tool-error';
    } else if (type === 'iteration-skipped') {
      it.isSkipped = true;
      it.skipReason = c?.reason === 'network_error' ? 'network_error' : 'same_inputs';
      it.modelResponse = it.skipReason === 'network_error'
        ? `Network error - iteration skipped: ${c?.error || 'Connection failed'}`
        : 'No significant change detected - iteration skipped to save resources.';
      this.finish(it, at);
    } else if (type === 'iteration-end') {
      this.finish(it, at);
    }
  }

  /** The one write: the iteration is complete, so persist it and trim memory. */
  private finish(it: IterationData, at: string) {
    it.endTime = at;
    it.duration = (Date.parse(at) - Date.parse(it.startTime)) / 1000;
    it.modelImageCount = it.modelImages?.length ?? 0;
    it.bytes = JSON.stringify(it).length;

    const text = textOf(it), images = it.modelImages ?? [];
    this.writes = this.writes.then(async () => {
      const tx = (await db()).transaction(['iterations', 'images'], 'readwrite');
      tx.objectStore('iterations').put(text);
      if (images.length) tx.objectStore('images').put({ id: it.id, images });
      await done(tx);
    }).catch(e => { Logger.error('IterationStore', `Failed to save iteration ${it.id}`, e); });

    // Only the last KEEP_IMAGES finished iterations keep their images in memory.
    const kept = this.withImages.get(it.agentId) ?? [];
    kept.push(it.id);
    while (kept.length > KEEP_IMAGES) delete this.live.get(kept.shift()!)?.modelImages;
    this.withImages.set(it.agentId, kept);

    // Finished after its session ended (its own code called stopAgent()): it lives in IndexedDB now.
    if (this.sessions.get(it.agentId)?.id !== it.sessionId) this.live.delete(it.id);
  }

  // ---- Sessions ----------------------------------------------------------------

  public startSession(agentId: string, sessionId: string) {
    this.sessions.set(agentId, { id: sessionId, count: 0 });
  }

  /** Nothing to save: every finished iteration is already in IndexedDB. In-flight ones finish on their own. */
  public async endSession(agentId: string): Promise<void> {
    const session = this.sessions.get(agentId);
    if (!session) return;
    this.sessions.delete(agentId);
    this.withImages.delete(agentId);
    for (const it of this.live.values()) if (it.sessionId === session.id && it.endTime) this.live.delete(it.id);
    await this.writes;
  }

  // ---- Sync reads (current session, from memory; the UI polls these) -----------

  public getIterationsForAgent(agentId: string): IterationData[] {
    const sessionId = this.sessions.get(agentId)?.id;
    if (!sessionId) return [];
    return [...this.live.values()]
      .filter(it => it.sessionId === sessionId)
      .sort((a, b) => a.sessionIterationNumber - b.sessionIterationNumber);
  }

  public getToolsFromLastIteration(agentId: string): ToolCall[] {
    return [...(this.getIterationsForAgent(agentId).at(-1)?.tools ?? [])];
  }

  // ---- Async reads (everything, from IndexedDB + memory) -----------------------

  /** One iteration WITH its images. */
  public async getIteration(id: string): Promise<IterationData | undefined> {
    await this.writes;
    const tx = (await db()).transaction(['iterations', 'images']);
    const [stored, img] = await Promise.all([
      req<IterationData | undefined>(tx.objectStore('iterations').get(id)),
      req<{ images: string[] } | undefined>(tx.objectStore('images').get(id)),
    ]);
    const it = this.live.get(id) ?? stored;
    return it && { ...it, modelImages: it.modelImages ?? img?.images ?? [] };
  }

  /** An agent's iterations across all sessions, newest first, text only unless `images`. */
  public async getRuns(
    agentId: string,
    { limit = 20, since, until, images = false }: { limit?: number; since?: string; until?: string; images?: boolean } = {},
  ): Promise<IterationData[]> {
    await this.writes;
    const index = (await db()).transaction('iterations').objectStore('iterations').index('agent_time');
    const stored: IterationData[] = [];
    await new Promise<void>((ok, fail) => {
      const cursor = index.openCursor(agentRange(agentId, since, until), 'prev');
      cursor.onsuccess = () => {
        const c = cursor.result;
        if (!c || stored.length >= limit) return ok();
        stored.push(c.value);
        c.continue();
      };
      cursor.onerror = () => fail(cursor.error);
    });

    // Memory wins: it has the in-flight iteration, which isn't stored yet.
    const byId = new Map(stored.map(it => [it.id, it]));
    for (const it of this.live.values()) {
      if (it.agentId === agentId && (!since || it.startTime >= since) && (!until || it.startTime <= until)) byId.set(it.id, it);
    }
    const runs = [...byId.values()].sort(newestFirst).slice(0, limit);
    if (!images) return runs.map(textOf);
    return Promise.all(runs.map(async it => (await this.getIteration(it.id)) ?? it));
  }

  /** Every session but the running one, newest first, iterations oldest first. Text only unless `images`. */
  public async getPastSessions(agentId: string, { images = false }: { images?: boolean } = {}): Promise<AgentSession[]> {
    const current = this.sessions.get(agentId)?.id;
    let runs = (await this.getRuns(agentId, { limit: Infinity })).filter(it => it.sessionId !== current);
    if (images) runs = await Promise.all(runs.map(async it => (await this.getIteration(it.id)) ?? it));

    const sessions = new Map<string, AgentSession>();
    for (const it of runs.reverse()) {
      let s = sessions.get(it.sessionId);
      if (!s) sessions.set(it.sessionId, s = { sessionId: it.sessionId, agentId, startTime: it.startTime, iterations: [] });
      s.iterations.push(it);
      s.endTime = it.endTime ?? s.endTime;
    }
    return [...sessions.values()].reverse();
  }

  // ---- Housekeeping ------------------------------------------------------------

  public async clearAllHistory(agentId: string): Promise<void> {
    for (const it of this.live.values()) if (it.agentId === agentId && it.endTime) this.live.delete(it.id);
    this.withImages.delete(agentId);
    const session = this.sessions.get(agentId);
    if (session) session.count = 0;

    await this.writes;
    const tx = (await db()).transaction(['iterations', 'images'], 'readwrite');
    const ids = await req(tx.objectStore('iterations').index('agent_time').getAllKeys(agentRange(agentId)));
    for (const id of ids) {
      tx.objectStore('iterations').delete(id);
      tx.objectStore('images').delete(id);
    }
    await done(tx);
  }

  public async getStorageUsage(agentId: string): Promise<{ currentSessionMB: number; totalHistoryMB: number }> {
    const runs = await this.getRuns(agentId, { limit: Infinity });
    const sessionId = this.sessions.get(agentId)?.id;
    const mb = (its: IterationData[]) => +(its.reduce((n, it) => n + (it.bytes ?? 0), 0) / 2 ** 20).toFixed(2);
    return { currentSessionMB: mb(runs.filter(it => it.sessionId === sessionId)), totalHistoryMB: mb(runs) };
  }

  public debug() {
    console.log('IterationStore', { sessions: [...this.sessions], live: [...this.live.values()] });
  }
}

export const IterationStore = new IterationStoreClass();
