# START Triage Protocol — Decision Flowchart

## Overview

START (Simple Triage And Rapid Treatment) is a triage system used by first responders at multiple-casualty or mass-casualty incidents. It sorts patients into four categories based on three rapid observations: **Respiration, Perfusion, and Mental Status (RPM)**. Each per-patient assessment should take 30 seconds or less and must never exceed one minute.

## Triage Categories (Tag Colors)

| Tag | Color | Meaning |
|---|---|---|
| MINOR | Green | Delayed care; can delay up to three hours |
| DELAYED | Yellow | Urgent care; can delay up to one hour |
| IMMEDIATE | Red | Immediate care required; life-threatening |
| DEAD | Black | Victim is deceased; no care required |

## Entry Point

Begin by starting where you stand. Assess the scene, call for assistance, and determine safety. Then call out to all victims.

## Step 1 — Ambulation Assessment (Call Out)

Call out instructions asking all victims who can walk to move to a specific designated safe area.

- **IF** the victim can walk (Walking Wounded & Uninjured) → Tag as **MINOR**. Hold them in a specific location and remember to fully triage them ASAP.
- **IF** the victim cannot walk (Non-Walking) → Proceed to Step 2 (Respiration Assessment).

Note: If a patient complains of pain on attempting to walk or move, do not force them to move.

## Step 2 — Respiration Assessment

Evaluate whether the patient is breathing.

### Branch A — Patient IS breathing

Determine the respiratory rate.

- **IF** breathing rate is **OVER 30 per minute** → Tag as **IMMEDIATE**.
- **IF** breathing rate is **UNDER 30 per minute** → Proceed to Step 3 (Perfusion Assessment).

### Branch B — Patient is NOT breathing

Position the airway using a head-tilt maneuver. Clear the mouth of foreign matter. Look, listen, and feel for breathing.

- **IF** the patient begins breathing after airway positioning → Tag as **IMMEDIATE**.
- **IF** the patient does NOT begin breathing → Reposition the airway and reassess (Look, Listen & Feel again).
  - **IF** the patient now breathes → Tag as **IMMEDIATE**.
  - **IF** the patient still does not breathe → Tag as **DEAD**.

Patients who need help maintaining an open airway are tagged IMMEDIATE. When in doubt about the patient's ability to breathe, tag as IMMEDIATE.

## Step 3 — Perfusion Assessment

Evaluate circulation using the radial pulse check and/or the blanch test (capillary refill). Either test alone is sufficient to make the triage decision at this step.

### Radial Pulse Test

Place index and middle fingers on the palm side of the wrist at the base of the thumb. Hold for 5–10 seconds.

- **IF** radial pulse is **ABSENT** (or irregular) → Tag as **IMMEDIATE**.
- **IF** radial pulse is **PRESENT** → Proceed to Step 4 (Mental Status).

### Blanch Test (Capillary Refill)

- **IF** capillary refill is **OVER 2 seconds** → Tag as **IMMEDIATE**.
- **IF** capillary refill is **UNDER 2 seconds** → Proceed to Step 4 (Mental Status).

## Step 4 — Mental Status Assessment

Reached only by patients with adequate breathing and adequate circulation. Give simple commands such as "Open your eyes," "Close your eyes," or "Squeeze my hand."

- **IF** the patient can **follow** simple commands → Tag as **DELAYED**.
- **IF** the patient **cannot follow** simple commands (unresponsive to verbal stimuli) → Tag as **IMMEDIATE**.

## Complete Decision Paths (All Possible Outcomes)

### Path to MINOR (Green)
1. Victim can walk on their own → **MINOR**

### Paths to IMMEDIATE (Red)
1. Non-walking → breathing rate over 30/min → **IMMEDIATE**
2. Non-walking → not breathing → airway positioned → begins breathing → **IMMEDIATE**
3. Non-walking → not breathing → airway repositioned → begins breathing → **IMMEDIATE**
4. Non-walking → breathing under 30/min → radial pulse absent → **IMMEDIATE**
5. Non-walking → breathing under 30/min → blanch test over 2 seconds → **IMMEDIATE**
6. Non-walking → breathing under 30/min → adequate perfusion → cannot follow simple commands → **IMMEDIATE**

### Path to DELAYED (Yellow)
1. Non-walking → breathing under 30/min → radial pulse present AND/OR blanch under 2 seconds → follows simple commands → **DELAYED**

### Path to DEAD (Black)
1. Non-walking → not breathing → airway positioned (no breath) → airway repositioned (still no breath) → **DEAD**

## Key Decision Thresholds

- **Respiratory rate threshold:** 30 breaths per minute (over → IMMEDIATE; under → continue assessment)
- **Capillary refill (blanch test) threshold:** 2 seconds (over → IMMEDIATE; under → continue assessment)
- **Radial pulse:** present (continue) or absent/irregular (IMMEDIATE)
- **Mental status:** follows commands (DELAYED) or cannot follow commands (IMMEDIATE)

## Key Principles

- Target assessment time per patient: 30 seconds or less; never exceed one minute per patient.
- When in doubt about a patient's ability to breathe, tag as IMMEDIATE.
- In mass-casualty situations, standard cervical spine stabilization protocols may be bypassed in order to open airways during triage.
- The goal of START is to rapidly identify IMMEDIATE patients for priority care.
- Re-triage patients as time and resources permit; patient conditions change, especially as shock progresses.
- Do not become involved in treating the first or second patient you encounter — move through all victims and tag them first.

## Node Reference (Flowchart Vocabulary)

- **START**: Entry node — Assess scene, call for assistance, determine safety.
- **Call Out**: Verbal instruction separating walking from non-walking victims.
- **RESPIRATION**: Breathing assessment node for non-walking victims.
- **PERFUSION**: Circulation assessment node (radial pulse + blanch test).
- **MENTAL STATUS**: Responsiveness / command-following assessment node.
- **Position Airway, Look, Listen & Feel**: First airway intervention for non-breathing patients.
- **Reposition Airway, Look, Listen & Feel**: Second airway attempt if the first fails.
