#!/usr/bin/env python3
"""Speech for the demos that agents listen to, made with macOS `say`.

Writes one .m4a per line under media/<script>/ and media/<script>.js, which sets
window.<NAME> to the lines with their durations, so a demo can play them back to back and
know when each one starts (file:// pages can't fetch JSON). Rerun after editing a script:

    python3 e2e/demos/media/make_audio.py
"""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

VOICES = {'Ana': 'Samantha', 'Ben': 'Daniel', 'Priya': 'Karen', 'Speaker': 'Reed (English (US))'}

# A weekly sync with three decisions and three owned action items, for the meeting scenarios.
# `hold` lines play once when the page loads, before the suite starts it.
MEETING = {
    'hold': [
        ('Ana', "Hi Ben. Let's give Priya a minute, she said she's running a bit late."),
        ('Ben', 'Sure, no rush.'),
    ],
    'lines': [
        ('Priya', "Hey everyone, sorry I'm late. The train was stuck outside the station."),
        ('Ana', "No worries. Okay, let's get started. Three things today: the launch date, pricing, and the beta."),
        ('Ana', "First, the launch date. Engineering asked for more time on the billing migration, so I'd like to move the launch from March third to March fourteenth. Any objections?"),
        ('Ben', 'None from me. Honestly, the extra eleven days would let us fix the Stripe webhook retries properly instead of patching them.'),
        ('Priya', 'Marketing can work with March fourteenth. It actually lines up better with the newsletter.'),
        ('Ana', "Great. So that's decided: the launch moves to March fourteenth."),
        ('Ana', "Second, pricing. Right now the annual plan has a twenty percent discount, and finance thinks that's too generous."),
        ('Priya', 'Our competitors are mostly at fifteen percent. I think fifteen is safe.'),
        ('Ben', 'As long as existing annual customers keep their current price until renewal.'),
        ('Ana', "Agreed. We'll drop the annual discount to fifteen percent, and existing customers keep their price until they renew."),
        ('Ana', "I'll update the pricing doc with the new numbers and send it around by Friday."),
        ('Ana', 'Last thing, the beta. We have about four hundred people on the waitlist. Do we open it up before launch?'),
        ('Ben', "I'd keep it invite only. The webhook issue could double charge people, and I don't want that hitting a thousand users."),
        ('Priya', "That's fair. If we keep it invite only, I can use the waitlist for the launch email instead."),
        ('Ana', 'Okay, decided: the beta stays invite only until launch.'),
        ('Ana', "So, action items. Ben, you'll fix the Stripe webhook retries?"),
        ('Ben', "Yes, I'll have the fix in staging by Wednesday."),
        ('Ana', "Priya, you'll draft the launch email?"),
        ('Priya', "Yep, I'll draft the launch email and share it with you both on Thursday."),
        ('Ana', "And I've got the pricing doc. Anything else before we wrap?"),
        ('Ben', "Just a reminder that I'm out next Monday."),
        ('Ana', 'Noted. Thanks everyone, see you next week.'),
    ],
}

# A conference talk with one slide per chapter, for the YouTube scenarios.
TALK = {
    'hold': [],
    'lines': [
        ('Why our cloud bill doubled',
         "Hi everyone, and thanks for coming. Last year our cloud bill doubled in about six months, while our traffic grew by only twenty percent. "
         "That was a scary chart to show the board. The strange part is that nobody had done anything wrong. "
         "There was no runaway script and no forgotten cluster. We just had hundreds of small decisions, made by different teams over a few years, "
         "and every one of them cost a little money. So in this talk I want to walk you through the four steps we took to cut the bill in half, "
         "what each step saved us, and what we would do differently if we started again today."),
        ('Step 1: Find where the money goes',
         "Step one is finding where the money actually goes, because you can't cut what you can't see. "
         "We turned on cost allocation tags and asked every team to tag its resources with a service name and an owner. "
         "Then we built a single dashboard in Cost Explorer, grouped by team and by service, and we reviewed it every Monday. "
         "Within the first week we found that forty percent of our spend had no owner at all. "
         "Some of it was old experiments, some of it was load balancers pointing at nothing, and some of it was snapshots from projects that ended years ago."),
        ('Step 2: Right-size your instances',
         "Step two is right-sizing. Most of our instances were picked years ago, by people who have since left, and nobody ever revisited them. "
         "We pulled two weeks of CPU and memory metrics for every instance, and anything that stayed under twenty percent of its capacity got moved one size down. "
         "We did it in small batches, watched latency for a day, and then moved on to the next batch. "
         "That alone saved eighteen percent of the whole bill, and our latency didn't change at all."),
        ('Step 3: Spot instances for batch jobs',
         "Step three is spot instances. A lot of our compute was nightly batch work: the data pipelines, the report builders, and the test runners for continuous integration. "
         "All of those jobs can survive being interrupted, as long as they retry. "
         "So we moved them to spot instances, added automatic retries with checkpoints, and spread them across several instance types. "
         "Spot capacity can be up to ninety percent cheaper than on demand, and in practice our batch costs dropped by about two thirds."),
        ('Step 4: Storage lifecycle rules',
         "Step four is storage. We kept every log file and every database backup forever, and all of it sat in the most expensive storage class. "
         "Now we have lifecycle rules on every bucket. Objects move to infrequent access after thirty days, and to Glacier after ninety days. "
         "Logs older than a year are deleted, unless legal asks us to keep them. "
         "Storage used to be our third biggest line item, and today it barely shows up on the dashboard."),
        ('Results: a 52% smaller bill',
         "So, putting it all together. Tagging showed us the waste, right-sizing and spot instances cut our compute, and lifecycle rules cut our storage. "
         "Our monthly bill went down by fifty two percent, and it has stayed there for three quarters now. "
         "The biggest lesson for us was that cost is a habit, not a project. Thanks for listening, and I'm happy to take questions."),
    ],
}


def make(name, script, voice_of):
    out_dir = os.path.join(HERE, name)
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for part in ('hold', 'lines'):
        rows = []
        for i, (who, text) in enumerate(script[part]):
            base = os.path.join(out_dir, f'{part}{i:02d}')
            subprocess.run(['say', '-v', voice_of(who), '-o', base + '.aiff', text], check=True)
            subprocess.run(['afconvert', '-f', 'm4af', '-d', 'aac', base + '.aiff', base + '.m4a'], check=True)
            info = subprocess.run(['afinfo', base + '.m4a'], check=True, capture_output=True, text=True).stdout
            duration = float(next(l for l in info.splitlines() if 'estimated duration' in l).split(':')[1].split()[0])
            os.remove(base + '.aiff')
            rows.append({'who': who, 'text': text, 'file': f'media/{name}/{part}{i:02d}.m4a', 'duration': round(duration, 2)})
        manifest[part] = rows
    with open(os.path.join(HERE, f'{name}.js'), 'w') as f:
        f.write(f'// Generated by make_audio.py: lines of speech and their durations in seconds.\n')
        f.write(f'window.{name.upper()} = {json.dumps(manifest, indent=1)};\n')
    total = sum(r['duration'] for r in manifest['lines'])
    print(f'{name}: {len(manifest["lines"])} lines, {total:.0f} s of speech')


make('meeting', MEETING, lambda who: VOICES[who])
make('talk', TALK, lambda who: VOICES['Speaker'])
