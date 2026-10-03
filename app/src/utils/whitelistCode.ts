// src/utils/whitelistCode.ts
//
// Generates the stable, per-user whitelist code shown in the golden-path QR
// (see UserInfoModal.tsx). It's a bearer value — whoever texts it becomes the
// bound phone number — so words are picked for low ambiguity (no homophones,
// no near-duplicates) rather than for memorability tricks that risk collisions.

export const WORDLIST = [
  'anchor', 'apple', 'arrow', 'autumn', 'badge', 'banjo', 'basil', 'beacon',
  'bicycle', 'binder', 'birch', 'bishop', 'blanket', 'blossom', 'bramble', 'brass',
  'breeze', 'bridge', 'bronze', 'bucket', 'bulb', 'cabin', 'candle', 'canvas',
  'canyon', 'cargo', 'carpet', 'cedar', 'cellar', 'chalk', 'charcoal', 'cherry',
  'chimney', 'cinder', 'clover', 'cobalt', 'compass', 'copper', 'coral', 'cotton',
  'crater', 'crayon', 'crescent', 'cricket', 'crimson', 'crystal', 'dagger', 'daisy',
  'dolphin', 'dragon', 'drift', 'drum', 'eagle', 'ember', 'engine', 'falcon',
  'feather', 'fennel', 'ferry', 'fiddle', 'fiesta', 'flame', 'flannel', 'flint',
  'forest', 'fossil', 'fountain', 'foxglove', 'frost', 'garden', 'garnet', 'gazelle',
  'ginger', 'glacier', 'goblet', 'granite', 'gravel', 'guitar', 'gully', 'hammer',
  'harbor', 'harvest', 'hazel', 'hearth', 'helmet', 'heron', 'hickory', 'holly',
  'hornet', 'hunter', 'iceberg', 'indigo', 'ivory', 'jacket', 'jasper', 'jigsaw',
  'jungle', 'kettle', 'kitten', 'lagoon', 'lantern', 'lark', 'lavender', 'ledger',
  'lemon', 'lentil', 'lighthouse', 'lilac', 'linen', 'lobster', 'locket', 'lumber',
  'magnet', 'mallard', 'mango', 'maple', 'marble', 'marigold', 'marsh', 'meadow',
  'mint', 'mitten', 'monsoon', 'mosaic', 'moss', 'mountain', 'mustang', 'nectar',
  'needle', 'nettle', 'nickel', 'nimbus', 'nutmeg', 'oak', 'oasis', 'obsidian',
  'olive', 'onyx', 'opal', 'orbit', 'orchard', 'osprey', 'otter', 'oxide',
  'paddle', 'palace', 'pansy', 'parcel', 'parsley', 'pebble', 'pelican', 'pepper',
  'petal', 'pheasant', 'pickle', 'pigeon', 'pillow', 'pilot', 'pine', 'pineapple',
  'pioneer', 'pocket', 'poplar', 'poppy', 'possum', 'pottery', 'prairie', 'pretzel',
  'puddle', 'pumpkin', 'quartz', 'quill', 'quilt', 'rabbit', 'raccoon', 'radish',
  'rainbow', 'raven', 'reed', 'ribbon', 'ripple', 'river', 'rocket', 'rooster',
  'rosemary', 'rubble', 'ruby', 'rustic', 'saddle', 'saffron', 'sage', 'salmon',
  'sandal', 'sapling', 'sapphire', 'satin', 'sawdust', 'scarf', 'shovel', 'shrimp',
  'sienna', 'silo', 'sketch', 'skylark', 'sleigh', 'sonnet', 'sparrow', 'spatula',
  'sphinx', 'spinach', 'spruce', 'squirrel', 'stable', 'starling', 'stone', 'stork',
  'stump', 'sunflower', 'swallow', 'syrup', 'tabby', 'tangerine', 'tapestry', 'tavern',
  'thistle', 'thunder', 'timber', 'toffee', 'tortoise', 'trellis', 'trumpet', 'tulip',
  'tundra', 'turnip', 'turquoise', 'turtle', 'tusk', 'umbrella', 'valley', 'velvet',
  'vessel', 'violet', 'walnut', 'walrus', 'warbler', 'wasp', 'wattle', 'whistle',
  'willow', 'windmill', 'wisteria', 'wombat', 'wrench', 'yarrow', 'yew', 'zephyr',
];

function randomIndex(max: number): number {
  const arr = new Uint32Array(1);
  crypto.getRandomValues(arr);
  return arr[0] % max;
}

/** A stable, human-typeable whitelist code, e.g. "tree-book-shower-golden". */
export function generateWhitelistCode(): string {
  const words = Array.from({ length: 4 }, () => WORDLIST[randomIndex(WORDLIST.length)]);
  return words.join('-');
}

/**
 * The two ways to pair a phone, each its own opt-in with its own code: texting the code to
 * Observer's SMS number (SMS + calls) or sending it on WhatsApp (WhatsApp + calls).
 */
export type PhoneChannel = 'sms' | 'whatsapp';

const WORDS = new Set(WORDLIST);

/**
 * The canonical code if `text` is one ("Tree book-shower golden" counts), else null.
 * Mirrors the server's remote.normalize_code: the phone tools only accept codes now,
 * so anything this rejects (e.g. a raw phone number) can never be sent to.
 */
export function normalizeWhitelistCode(text: string | null | undefined): string | null {
  if (!text) return null;
  const parts = text.trim().toLowerCase().split(/[\s-]+/).filter(Boolean);
  return parts.length === 4 && parts.every(p => WORDS.has(p)) ? parts.join('-') : null;
}
