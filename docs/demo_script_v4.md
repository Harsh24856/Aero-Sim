# AERO-SIM demo script: physics v4 (7 minutes)

This is for the PS 26054 judges. The v3 script is `docs/demo_script.md`.

**The story:** a real-physics Rotax engine; a healthy twin flying beside it on board; an AI that names the failing part, tells a broken sensor from a broken engine, and counts down the hours left. The injected truth is on screen the whole time, so the audience can check the AI.

Every number to say aloud is in `docs/model_cards_v4.md`. Don't quote anything that isn't in there.

## Before the room (10 minutes before)

1. **Start the stack:** `AERO_PHYSICS_VERSION=v4 scripts/start_stack.sh`, then wait for "Open http://localhost:3000". Use `localhost`, not `127.0.0.1`.
2. **Open the cockpit:** sign in and open `/simulate?engine=Rotax_914_ULF`. The **Engine** strip above the view should list the demo scenarios, with the life-scale label "×180 life · 1 real min = 3 engine h".
3. **Close everything else:** the laptop has 8 GB, and the AI runs on the GPU.
4. **Backup:** the recorded video on the desktop, plus `docs/model_cards_v4.md` open in a tab.

## 0:00 – The problem (30 s)

"A MALE UAV engine failure ends the mission, and often the aircraft. DRDO asked for a digital twin that watches a Rotax-class engine live, names what is failing, and tells a maintainer how long the engine has left. We built one for four real Rotax engines."

## 0:30 – The twin (60 s)

- **Start it:** choose **Healthy engine, 300 h** and press **Start**.
- **Point at "Engine vs healthy twin":** "Orange is what each sensor reads. The dashed line is a perfectly healthy copy of this engine, flown on the same throttle, altitude and weather, on board, every second. On a healthy engine they lie on top of each other."
- **The two clocks:** "Flight time is real seconds, because that's what the AI was trained on. Engine hours run 180 times faster, so we can watch an engine age during a demo. That's one declared constant, and every run records it."
- **When the AI comes online (128 s):** it reads **nominal**, its wear condition sits close to the true value in the ground-truth panel, and **RUL equals the calendar countdown**. That's correct: a healthy engine will reach its scheduled overhaul. In the end-to-end check it was 1,681 h, equal to both the truth and the calendar.

## 1:30 – A failing engine (90 s)

- **Stop, choose Bearing wear developing, 1,450 h, and Start.**
- **The twin chart:** oil pressure separates from the twin. "Worn bearings open their clearances, so the pump can't hold pressure."
- **"Which part?"** names **Bearing wear** with its severity. The **ground-truth panel** below says the same thing. "The AI never sees that panel. It worked this out from the sensors."
- **The advisory:** "Take an oil sample for analysis and inspect the oil filter for metal." That's the real Rotax maintenance action.

## 3:00 – Engine, or sensor? (60 s)

- **Mid-flight, break a sensor:** in the **Inject** strip, choose **CHT, dropout**, then **Break sensor**. Use a dropout, not "stuck": dropout was recognised on 56 of 57 samples in the end-to-end check, and a stuck EGT on none.
- **What changes:** the CHT tile turns to **dropout**, and the CHT line on the twin chart greys out.
- **What doesn't change:** the engine fault list. "A failed thermocouple must not ground a healthy engine. The advisory says to check the sender, not the engine."

## 4:00 – A hot day, and a tired radiator (60 s)

- **Stop, choose Cooling degradation on a hot day (ISA +20), and Start.**
- **The twin chart:** CHT and oil temperature sit above the healthy twin. "The twin flies the same hot day, so the heat of the day cancels out. What's left is the radiator."
- **"Which part?"** names **Cooling degradation**. The advisory says to check coolant level, radiator, hoses and airflow, and, if it gets worse, to raise airspeed and reduce power now.
- **If asked about turbo faults:** "Its probability bar climbs, but it stays under its cut-off, and the model often names another fault that starves the engine of air instead. Below 15,000 ft a real 914 hides turbo faults completely. We show that as it is." (The numbers are in the Q&A notes.)

## 5:00 – Evidence, not claims (45 s)

Show `docs/model_cards_v4.md`. All figures are on flights the models never saw:
- **Detection:** AUC **0.879**, and **0.952** where the engine has drifted from its twin.
- **Naming the failing part:** F1 **0.612** over 13 faults. Bearing wear 0.89, cooling 0.87, fuel metering 0.96.
- **Wear condition:** within **6 points out of 100**.
- **Remaining life, said plainly:** "On engines that wear out early, it's off by 387 h, where the overhaul calendar is off by 462 h. That's better than the calendar, but not yet our 15% target. We report the number an aircraft would really get, not the better one we could have shown."
- **Proof the live system matches training:** the live window is **identical** to training, with zero difference on real test flights. An independent Simulink model matches the physics to **2e-15**.

## 5:45 – Maintainer view (45 s)

- **Mission report:** stop, and open the mission report. It shows the engine hours at the start and end of the flight, the life scale, and the post-flight summary.
- **Replay:** scrub to any second and see **what the AI named** next to **what was really injected**.
- **Dashboard → Engines:** "Every engine keeps its hour meter and its maintenance history between flights."

## 6:30 – Close (30 s)

"Real-physics twin, six models per engine, honest numbers, and a plan to calibrate on real engine data. The limits are documented: sensor drift needs a longer view than two minutes, and remaining life needs better wear estimates. What we need next is bench data from a Rotax-class engine."

## If something goes wrong

| Symptom | Say / do |
|---|---|
| The cockpit shows "Placeholder AI" on a 912/915/916 | That engine is still using the 914's models; demo on the 914. |
| "AI service unavailable" | "Physics, the twin chart and the limit checks keep running without the AI." Carry on. |
| The engine strip doesn't appear | The backend isn't on v4: restart with `AERO_PHYSICS_VERSION=v4`. |
| The page won't load | Play the backup video from the same point. |
