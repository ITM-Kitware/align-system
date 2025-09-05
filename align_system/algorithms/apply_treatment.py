import argparse
import json
import sys
import yaml
from pathlib import Path

from align_system.data_models.compat.ta3_ph1_client_models import (
    InjuryLocationEnum,
    InjurySeverityEnum,
    InjuryStatusEnum,
    InjuryTypeEnum,
    SupplyTypeEnum,
)
from align_system.utils import logging


_logger = logging.getLogger(__name__)


def _have_supply(name: str, supplies: list) -> bool:
    for supply in supplies:
        if supply["type"] == name:
            return supply["quantity"] > 0
    return False


def _supplies_to_str(supplies: list, prefix: str = "") -> str:
    string = ""
    for supply in supplies:
        if supply["quantity"] > 0:
            string += f"{prefix}{supply['quantity']} x {supply['type']}\n"
    return string


def _is_face(body_location: str) -> bool:
    if body_location == InjuryLocationEnum.LEFT_FACE or \
            body_location == InjuryLocationEnum.RIGHT_FACE:
        return True
    return False


def _is_limb(body_location: str) -> bool:
    if body_location == InjuryLocationEnum.RIGHT_FOREARM or \
       body_location == InjuryLocationEnum.LEFT_FOREARM or \
       body_location == InjuryLocationEnum.RIGHT_HAND or \
       body_location == InjuryLocationEnum.LEFT_HAND or \
       body_location == InjuryLocationEnum.RIGHT_LEG or \
       body_location == InjuryLocationEnum.LEFT_LEG or \
       body_location == InjuryLocationEnum.RIGHT_CALF or \
       body_location == InjuryLocationEnum.LEFT_CALF or \
       body_location == InjuryLocationEnum.RIGHT_THIGH or \
       body_location == InjuryLocationEnum.LEFT_THIGH or \
       body_location == InjuryLocationEnum.RIGHT_BICEP or \
       body_location == InjuryLocationEnum.LEFT_BICEP or \
       body_location == InjuryLocationEnum.RIGHT_WRIST or \
       body_location == InjuryLocationEnum.LEFT_WRIST:
        return True

    return False


def _tourniquet_location(injury_location: str):
    # This assumes the location is a limb
    if injury_location == InjuryLocationEnum.RIGHT_HAND or \
       injury_location == InjuryLocationEnum.RIGHT_WRIST:
        return InjuryLocationEnum.RIGHT_FOREARM
    elif injury_location == InjuryLocationEnum.LEFT_HAND or \
            injury_location == InjuryLocationEnum.LEFT_WRIST:
        return InjuryLocationEnum.LEFT_FOREARM
    return injury_location


def _apply_treatment(injury_location: str, supplies: list, supply_precedence: list) -> (str, str):
    for supply in supply_precedence:

        if supply == SupplyTypeEnum.BLANKET and _have_supply(SupplyTypeEnum.BLANKET, supplies):
            return SupplyTypeEnum.BLANKET, injury_location

        if supply == SupplyTypeEnum.BLOOD and _have_supply(SupplyTypeEnum.BLOOD, supplies):
            # TODO ensure the infusion site is not injured (Either Bicep)
            return SupplyTypeEnum.BLOOD, InjuryLocationEnum.RIGHT_BICEP

        if supply == SupplyTypeEnum.BURN_DRESSING and _have_supply(SupplyTypeEnum.BURN_DRESSING, supplies):
            return SupplyTypeEnum.BURN_DRESSING, injury_location

        if supply == SupplyTypeEnum.DECOMPRESSION_NEEDLE and _have_supply(SupplyTypeEnum.DECOMPRESSION_NEEDLE, supplies):
            return SupplyTypeEnum.DECOMPRESSION_NEEDLE, injury_location

        if supply == SupplyTypeEnum.EPI_PEN and _have_supply(SupplyTypeEnum.EPI_PEN, supplies):
            # TODO ensure the infusion site is not injured (Either Thigh or shoulder)
            return SupplyTypeEnum.EPI_PEN, InjuryLocationEnum.RIGHT_THIGH

        if supply == SupplyTypeEnum.FENTANYL_LOLLIPOP and _have_supply(SupplyTypeEnum.FENTANYL_LOLLIPOP, supplies):
            return SupplyTypeEnum.PAIN_MEDICATIONS, InjuryLocationEnum.HEAD

        if supply == SupplyTypeEnum.HEMOSTATIC_GAUZE and _have_supply(SupplyTypeEnum.HEMOSTATIC_GAUZE, supplies):
            return SupplyTypeEnum.HEMOSTATIC_GAUZE, injury_location

        if supply == SupplyTypeEnum.IV_BAG and _have_supply(SupplyTypeEnum.IV_BAG, supplies):
            # TODO ensure the infusion site is not injured
            return SupplyTypeEnum.IV_BAG, InjuryLocationEnum.RIGHT_BICEP

        if supply == SupplyTypeEnum.NASOPHARYNGEAL_AIRWAY and _have_supply(SupplyTypeEnum.NASOPHARYNGEAL_AIRWAY, supplies):
            return SupplyTypeEnum.NASOPHARYNGEAL_AIRWAY, InjuryLocationEnum.HEAD

        if supply == SupplyTypeEnum.PAIN_MEDICATIONS and _have_supply(SupplyTypeEnum.PAIN_MEDICATIONS, supplies):
            return SupplyTypeEnum.PAIN_MEDICATIONS, InjuryLocationEnum.HEAD

        if supply == SupplyTypeEnum.PRESSURE_BANDAGE and _have_supply(SupplyTypeEnum.PRESSURE_BANDAGE, supplies):
            return SupplyTypeEnum.PRESSURE_BANDAGE, injury_location

        if supply == SupplyTypeEnum.PULSE_OXIMETER and _have_supply(SupplyTypeEnum.PULSE_OXIMETER, supplies):
            # TODO ensure the infusion site is not injured
            return SupplyTypeEnum.PULSE_OXIMETER, InjuryLocationEnum.RIGHT_HAND

        if supply == SupplyTypeEnum.SPLINT and _have_supply(SupplyTypeEnum.SPLINT, supplies):
            return SupplyTypeEnum.SPLINT, injury_location

        if supply == SupplyTypeEnum.TOURNIQUET and _have_supply(SupplyTypeEnum.TOURNIQUET, supplies):
            return SupplyTypeEnum.TOURNIQUET, _tourniquet_location(injury_location)

        if supply == SupplyTypeEnum.VENTED_CHEST_SEAL and _have_supply(SupplyTypeEnum.VENTED_CHEST_SEAL, supplies):
            return SupplyTypeEnum.VENTED_CHEST_SEAL, injury_location

    # _logger.warning("No supplies to treat provided injury location")
    return "", ""


def treat_abrasion(injury_location: str, injury_severity: str, supplies: list):
    treatment = {}
    if injury_severity == InjurySeverityEnum.MINOR or injury_severity == InjurySeverityEnum.MODERATE:
        treatment["treatment"] = "supply"
        treatment["location"] = "head"
    else:
        treatment["treatment"] = "supply"
        treatment["location"] = "head"
    return treatment


def recommend_treatments(injuries: list, supplies: list) -> list:
    opts = []

    for injury in injuries:

        injury_treatment = {"consider": False,
                            "injury": "",
                            "treatment": "",
                            "parameters": None,
                            "issue": ""
                            }
        opts.append(injury_treatment)
        injury_type = injury["name"]
        injury_location = injury["location"]
        injury_severity = injury["severity"]
        injury_treatment["injury"] = (f"Found a {injury['status']} {injury['severity']}"
                                      f" {injury['name']} on the {injury['location']}.")
        if injury["status"] == InjuryStatusEnum.TREATED:
            injury_treatment["issue"] = "Injury has been treated"
            continue

        if injury["status"] == InjuryStatusEnum.VISIBLE or \
           injury["status"] == InjuryStatusEnum.DISCOVERED or \
           injury["status"] == InjuryStatusEnum.PARTIALLY_TREATED:
            injury_treatment["consider"] = True

        if injury_location == InjuryLocationEnum.UNSPECIFIED:
            # TODO  Discuss with group how to handle this case
            injury_treatment["issue"] = "No injury location specified"
            continue

        def treat():
            injury_treatment["treatment"] = f"Treating with a {supply}" \
                                            f" applied to the {location}."
            injury_treatment["parameters"] = {"treatment": supply, "location": location}

        """ LACERATION, PUNCTURE, SHRAPNEL """
        if injury_type == InjuryTypeEnum.LACERATION or \
                injury_type == InjuryTypeEnum.PUNCTURE or \
                injury_type == InjuryTypeEnum.SHRAPNEL:
            if injury_location == InjuryLocationEnum.INTERNAL:
                injury_treatment["issue"] = f"{injury_location} {injury_type} is not supported"
                injury_treatment["consider"] = False
                continue

            if _is_limb(injury_location):
                if injury_severity == InjurySeverityEnum.MINOR or injury_severity == InjurySeverityEnum.MODERATE:
                    supply, location = _apply_treatment(injury_location, supplies,
                                                        supply_precedence=[SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                           SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                           SupplyTypeEnum.TOURNIQUET,
                                                                           SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                           SupplyTypeEnum.FENTANYL_LOLLIPOP,
                                                                           SupplyTypeEnum.BLANKET])
                else:
                    supply, location = _apply_treatment(injury_location, supplies,
                                                        supply_precedence=[SupplyTypeEnum.TOURNIQUET,
                                                                           SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                           SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                           SupplyTypeEnum.BLOOD,
                                                                           SupplyTypeEnum.IV_BAG,
                                                                           SupplyTypeEnum.BLANKET])
                treat()
                continue

            elif injury_location == InjuryLocationEnum.LEFT_CHEST or \
                    injury_location == InjuryLocationEnum.RIGHT_CHEST or \
                    injury_location == InjuryLocationEnum.CENTER_CHEST or \
                    injury_location == InjuryLocationEnum.LEFT_SIDE or \
                    injury_location == InjuryLocationEnum.RIGHT_SIDE:
                if injury_severity == InjurySeverityEnum.MINOR or injury_severity == InjurySeverityEnum.MODERATE:
                    supply, location = _apply_treatment(injury_location, supplies,
                                                        supply_precedence=[SupplyTypeEnum.VENTED_CHEST_SEAL,
                                                                           SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                           SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                           SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                           SupplyTypeEnum.FENTANYL_LOLLIPOP])
                else:
                    supply, location = _apply_treatment(injury_location, supplies,
                                                        supply_precedence=[SupplyTypeEnum.VENTED_CHEST_SEAL,
                                                                           SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                           SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                           SupplyTypeEnum.BLOOD,
                                                                           SupplyTypeEnum.IV_BAG,
                                                                           SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                           SupplyTypeEnum.FENTANYL_LOLLIPOP])
                treat()
                continue

            elif injury_location == InjuryLocationEnum.LEFT_FACE or injury_location == InjuryLocationEnum.RIGHT_FACE:
                if injury_severity == InjurySeverityEnum.MINOR or injury_severity == InjurySeverityEnum.MODERATE:
                    supply, location = _apply_treatment(injury_location, supplies,
                                                        supply_precedence=[SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                           SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                           SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                           SupplyTypeEnum.FENTANYL_LOLLIPOP])
                else:
                    supply, location = _apply_treatment(injury_location, supplies,
                                                        supply_precedence=[SupplyTypeEnum.NASOPHARYNGEAL_AIRWAY,
                                                                           SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                           SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                           SupplyTypeEnum.BLOOD,
                                                                           SupplyTypeEnum.IV_BAG,
                                                                           SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                           SupplyTypeEnum.FENTANYL_LOLLIPOP])
                treat()
                continue

            else:
                if injury_severity == InjurySeverityEnum.MINOR or injury_severity == InjurySeverityEnum.MODERATE:
                    supply, location = _apply_treatment(injury_location, supplies,
                                                        supply_precedence=[SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                           SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                           SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                           SupplyTypeEnum.FENTANYL_LOLLIPOP])
                else:
                    supply, location = _apply_treatment(injury_location, supplies,
                                                        supply_precedence=[SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                           SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                           SupplyTypeEnum.BLOOD,
                                                                           SupplyTypeEnum.IV_BAG,
                                                                           SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                           SupplyTypeEnum.FENTANYL_LOLLIPOP])
                treat()
                continue

        """ ABRASION """
        if injury_type == InjuryTypeEnum.ABRASION:
            if injury_location == InjuryLocationEnum.INTERNAL:
                injury_treatment["issue"] = f"{injury_location} {injury_type} is not supported"
                injury_treatment["consider"] = False
                continue

            if injury_severity == InjurySeverityEnum.MINOR or injury_severity == InjurySeverityEnum.MODERATE:
                supply, location = _apply_treatment(injury_location, supplies,
                                                    supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS])
            else:
                supply, location = _apply_treatment(injury_location, supplies,
                                                    supply_precedence=[SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                       SupplyTypeEnum.PAIN_MEDICATIONS])
            treat()
            continue

        """ AMPUTATION """
        if injury_type == InjuryTypeEnum.AMPUTATION:
            if not _is_limb(injury_location):
                injury_treatment["issue"] = f"{injury_location} {injury_type} is not supported"
                injury_treatment["consider"] = False
                continue
            # Assuming that there is still enough of the injured limb left to be able to apply a tourniquet on that site
            supply, location = _apply_treatment(injury_location, supplies,
                                                supply_precedence=[SupplyTypeEnum.TOURNIQUET,
                                                                   SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                   SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                   SupplyTypeEnum.FENTANYL_LOLLIPOP,
                                                                   SupplyTypeEnum.PAIN_MEDICATIONS])
            treat()
            continue

        """ ASTHMATIC """
        if injury_type == InjuryTypeEnum.ASTHMATIC:
            # TODO What location are we expecting for this?
            # Assuming that there is still enough of the injured limb left to be able to apply a tourniquet on that site
            supply, location = _apply_treatment(injury_location, supplies,
                                                supply_precedence=[SupplyTypeEnum.EPI_PEN,
                                                                   SupplyTypeEnum.IV_BAG])
            treat()
            continue

        """ BROKEN_BONE """
        if injury_type == InjuryTypeEnum.BROKEN_BONE:
            if injury_location == InjuryLocationEnum.STOMACH or \
                    injury_location == InjuryLocationEnum.LEFT_STOMACH or \
                    injury_location == InjuryLocationEnum.RIGHT_STOMACH or \
                    injury_location == InjuryLocationEnum.INTERNAL:
                injury_treatment["issue"] = f"{injury_location} {injury_type} is not supported"
                injury_treatment["consider"] = False
                continue

            if _is_limb(injury_location) or \
                    injury_location == InjuryLocationEnum.LEFT_SHOULDER or \
                    injury_location == InjuryLocationEnum.RIGHT_SHOULDER or \
                    injury_location == InjuryLocationEnum.NECK or \
                    injury_location == InjuryLocationEnum.LEFT_NECK or \
                    injury_location == InjuryLocationEnum.RIGHT_NECK:
                supply, location = _apply_treatment(injury_location, supplies,
                                                    supply_precedence=[SupplyTypeEnum.SPLINT,
                                                                       SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                       SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                       SupplyTypeEnum.FENTANYL_LOLLIPOP])
            else:
                supply, location = _apply_treatment(injury_location, supplies,
                                                    supply_precedence=[SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                       SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                       SupplyTypeEnum.FENTANYL_LOLLIPOP])

            treat()
            continue

        """ BURN """
        if injury_type == InjuryTypeEnum.BURN:
            if injury_location == InjuryLocationEnum.INTERNAL:
                _logger.error(f"{injury_location} {injury_type} is not supported")
                injury_treatment["issue"] = f"{injury_location} {injury_type} is not supported"
                injury_treatment["consider"] = False
                continue
            if _is_face(injury_location):
                if injury_severity == InjurySeverityEnum.MINOR or injury_severity == InjurySeverityEnum.MODERATE:
                    supply, location = _apply_treatment(injury_location, supplies,
                                                        supply_precedence=[SupplyTypeEnum.BURN_DRESSING,
                                                                           SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                           SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                           SupplyTypeEnum.FENTANYL_LOLLIPOP])
                else:
                    supply, location = _apply_treatment(injury_location, supplies,
                                                        supply_precedence=[SupplyTypeEnum.NASOPHARYNGEAL_AIRWAY,
                                                                           SupplyTypeEnum.BURN_DRESSING,
                                                                           SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                           SupplyTypeEnum.IV_BAG,
                                                                           SupplyTypeEnum.BLOOD,
                                                                           SupplyTypeEnum.FENTANYL_LOLLIPOP,
                                                                           SupplyTypeEnum.PAIN_MEDICATIONS])
            else:
                supply, location = _apply_treatment(injury_location, supplies,
                                                    supply_precedence=[SupplyTypeEnum.BURN_DRESSING,
                                                                       SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                       SupplyTypeEnum.FENTANYL_LOLLIPOP,
                                                                       SupplyTypeEnum.PAIN_MEDICATIONS])
            treat()
            continue

        """ CHEST_COLLAPSE (Assuming a Closed Pneumothorax) """
        if injury_type == InjuryTypeEnum.CHEST_COLLAPSE:
            if injury_location != InjuryLocationEnum.LEFT_CHEST and \
                    injury_location != InjuryLocationEnum.LEFT_SIDE and \
                    injury_location != InjuryLocationEnum.RIGHT_CHEST and \
                    injury_location != InjuryLocationEnum.RIGHT_SIDE:
                injury_treatment["issue"] = f"{injury_location} {injury_type} is not supported"
                injury_treatment["consider"] = False
                continue
            supply, location = _apply_treatment(injury_location, supplies,
                                                supply_precedence=[SupplyTypeEnum.DECOMPRESSION_NEEDLE,
                                                                   SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                   SupplyTypeEnum.FENTANYL_LOLLIPOP])
            treat()
            continue

        """ EAR_BLEED """
        if injury_type == InjuryTypeEnum.EAR_BLEED:
            if injury_location != InjuryLocationEnum.HEAD and \
                    injury_location != InjuryLocationEnum.LEFT_FACE and \
                    injury_location != InjuryLocationEnum.RIGHT_FACE:
                injury_treatment["issue"] = f"{injury_location} {injury_type} is not supported"
                injury_treatment["consider"] = False
                continue

            if injury_severity == InjurySeverityEnum.MINOR or injury_severity == InjurySeverityEnum.MODERATE:
                supply, location = _apply_treatment(injury_location, supplies,
                                                    supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                       SupplyTypeEnum.FENTANYL_LOLLIPOP])
            else:
                supply, location = _apply_treatment(injury_location, supplies,
                                                    supply_precedence=[SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                       SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                       SupplyTypeEnum.FENTANYL_LOLLIPOP,
                                                                       SupplyTypeEnum.PAIN_MEDICATIONS])

            treat()
            continue

        """ INTERNAL """
        if injury_type == InjuryTypeEnum.INTERNAL:
            if injury_location != InjuryLocationEnum.INTERNAL:
                injury_treatment["issue"] = f"{injury_location} {injury_type} is not supported"
                injury_treatment["consider"] = False
                continue

            supply, location = _apply_treatment(injury_location, supplies,
                                                supply_precedence=[SupplyTypeEnum.IV_BAG,
                                                                   SupplyTypeEnum.BLOOD,
                                                                   SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                   SupplyTypeEnum.FENTANYL_LOLLIPOP])

            treat()
            continue

        """ OPEN ABDOMINAL WOUND """
        if injury_type == InjuryTypeEnum.OPEN_ABDOMINAL_WOUND:
            if injury_location != InjuryLocationEnum.STOMACH and \
                    injury_location != InjuryLocationEnum.LEFT_STOMACH and \
                    injury_location != InjuryLocationEnum.RIGHT_STOMACH:
                injury_treatment["issue"] = f"{injury_location} {injury_type} is not supported"
                injury_treatment["consider"] = False
                continue

            if injury_severity == InjurySeverityEnum.MINOR or injury_severity == InjurySeverityEnum.MODERATE:
                supply, location = _apply_treatment(injury_location, supplies,
                                                    supply_precedence=[SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                       SupplyTypeEnum.PAIN_MEDICATIONS])
            else:
                supply, location = _apply_treatment(injury_location, supplies,
                                                    supply_precedence=[SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                       SupplyTypeEnum.IV_BAG,
                                                                       SupplyTypeEnum.BLOOD,
                                                                       SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                       SupplyTypeEnum.FENTANYL_LOLLIPOP])

            treat()
            continue

        """ TRAUMATIC BRAIN INJURY """
        if injury_type == InjuryTypeEnum.TRAUMATIC_BRAIN_INJURY:
            if injury_location != InjuryLocationEnum.HEAD:
                injury_treatment["issue"] = f"{injury_location} {injury_type} is not supported"
                injury_treatment["consider"] = False
                continue

            if injury_severity == InjurySeverityEnum.MINOR or injury_severity == InjurySeverityEnum.MODERATE:
                supply, location = _apply_treatment(injury_location, supplies,
                                                    supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                       SupplyTypeEnum.FENTANYL_LOLLIPOP])
            else:
                supply, location = _apply_treatment(injury_location, supplies,
                                                    supply_precedence=[SupplyTypeEnum.HEMOSTATIC_GAUZE,
                                                                       SupplyTypeEnum.PRESSURE_BANDAGE,
                                                                       SupplyTypeEnum.PAIN_MEDICATIONS,
                                                                       SupplyTypeEnum.FENTANYL_LOLLIPOP,
                                                                       SupplyTypeEnum.IV_BAG,
                                                                       SupplyTypeEnum.BLOOD])

            treat()
            continue

    return opts


def treatment_options(injuries: list, supplies: list) -> dict:
    opts = {"injuries": [],
            "treatments": [],
            "parameters": [],
            "issues": []
            }

    for injury in injuries:
        injury_status = injury['status']
        injury_type = injury["name"]
        injury_location = injury["location"]
        injury_severity = injury["severity"]
        injury_description = f"Found a {injury_status} {injury_severity} {injury_type} on the {injury_location}."
        if injury_description not in opts["injuries"]:
            opts["injuries"].append(injury_description)

        if injury_status == InjuryStatusEnum.HIDDEN or \
                injury_status == InjuryStatusEnum.DISCOVERABLE or \
                injury_status == InjuryStatusEnum.TREATED:
            continue

        if injury_location == InjuryLocationEnum.UNSPECIFIED:
            # TODO  Discuss with group how to handle this case
            invalid_location = "No injury location specified"
            if invalid_location not in opts["issues"]:
                opts["issues"].append("No injury location specified")
            continue

        def recommend(supply_location):
            if not supply_location[0] or not supply_location[1]:
                return

            rec = (f"Treat the {injury_severity} {injury_type} of the {injury_location} "
                   f"with a {supply_location[0]} applied to the {supply_location[1]}.")
            if rec not in opts["treatments"]:
                opts["treatments"].append(rec)
                opts["parameters"].append({"treatment": supply_location[0], "location": supply_location[1]})

        """ LACERATION, PUNCTURE, SHRAPNEL """
        if injury_type == InjuryTypeEnum.LACERATION or \
                injury_type == InjuryTypeEnum.PUNCTURE or \
                injury_type == InjuryTypeEnum.SHRAPNEL:
            if injury_location == InjuryLocationEnum.INTERNAL:
                issue = f"{injury_location} {injury_type} is not supported"
                if issue not in opts["issues"]:
                    opts["issues"].append(issue)
                continue

            if _is_limb(injury_location):
                recommend(_apply_treatment(injury_location, supplies,
                                           supply_precedence=[SupplyTypeEnum.TOURNIQUET]))

            elif injury_location == InjuryLocationEnum.LEFT_CHEST or \
                    injury_location == InjuryLocationEnum.RIGHT_CHEST or \
                    injury_location == InjuryLocationEnum.CENTER_CHEST or \
                    injury_location == InjuryLocationEnum.LEFT_SIDE or \
                    injury_location == InjuryLocationEnum.RIGHT_SIDE:
                recommend(_apply_treatment(injury_location, supplies,
                                           supply_precedence=[SupplyTypeEnum.VENTED_CHEST_SEAL]))

            elif (injury_location == InjuryLocationEnum.LEFT_FACE or
                  injury_location == InjuryLocationEnum.RIGHT_FACE):
                recommend(_apply_treatment(injury_location, supplies,
                                           supply_precedence=[SupplyTypeEnum.NASOPHARYNGEAL_AIRWAY]))

            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PRESSURE_BANDAGE]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.HEMOSTATIC_GAUZE]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.FENTANYL_LOLLIPOP]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.BLOOD]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.IV_BAG]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.BLANKET]))
            continue

        """ ABRASION """
        if injury_type == InjuryTypeEnum.ABRASION:
            if injury_location == InjuryLocationEnum.INTERNAL:
                issue = f"{injury_location} {injury_type} is not supported"
                if issue not in opts["issues"]:
                    opts["issues"].append(issue)
                continue

            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PRESSURE_BANDAGE]))
            continue

        """ AMPUTATION """
        if injury_type == InjuryTypeEnum.AMPUTATION:
            if not _is_limb(injury_location):
                issue = f"{injury_location} {injury_type} is not supported"
                if issue not in opts["issues"]:
                    opts["issues"].append(issue)
                continue

            # Assuming there is still enough of the injured limb left to be able to apply a tourniquet on that site
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.TOURNIQUET]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.HEMOSTATIC_GAUZE]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PRESSURE_BANDAGE]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.BLOOD]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.IV_BAG]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.FENTANYL_LOLLIPOP]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.BLANKET]))
            continue

        """ ASTHMATIC """
        if injury_type == InjuryTypeEnum.ASTHMATIC:
            # TODO What location are we expecting for this?

            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.EPI_PEN]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.IV_BAG]))
            continue

        """ BROKEN_BONE """
        if injury_type == InjuryTypeEnum.BROKEN_BONE:
            if injury_location == InjuryLocationEnum.STOMACH or \
                    injury_location == InjuryLocationEnum.LEFT_STOMACH or \
                    injury_location == InjuryLocationEnum.RIGHT_STOMACH or \
                    injury_location == InjuryLocationEnum.INTERNAL:
                issue = f"{injury_location} {injury_type} is not supported"
                if issue not in opts["issues"]:
                    opts["issues"].append(issue)
                continue

            if _is_limb(injury_location) or \
                    injury_location == InjuryLocationEnum.LEFT_SHOULDER or \
                    injury_location == InjuryLocationEnum.RIGHT_SHOULDER or \
                    injury_location == InjuryLocationEnum.NECK or \
                    injury_location == InjuryLocationEnum.LEFT_NECK or \
                    injury_location == InjuryLocationEnum.RIGHT_NECK:
                recommend(_apply_treatment(injury_location, supplies,
                                           supply_precedence=[SupplyTypeEnum.SPLINT]))

            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PRESSURE_BANDAGE]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.FENTANYL_LOLLIPOP]))
            continue

        """ BURN """
        if injury_type == InjuryTypeEnum.BURN:
            if injury_location == InjuryLocationEnum.INTERNAL:
                _logger.error(f"{injury_location} {injury_type} is not supported")
                issue = f"{injury_location} {injury_type} is not supported"
                if issue not in opts["issues"]:
                    opts["issues"].append(issue)
                continue

            if _is_face(injury_location):
                recommend(_apply_treatment(injury_location, supplies,
                                           supply_precedence=[SupplyTypeEnum.NASOPHARYNGEAL_AIRWAY]))

            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.BURN_DRESSING]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PRESSURE_BANDAGE]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.IV_BAG]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.BLOOD]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.FENTANYL_LOLLIPOP]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS]))
            continue

        """ CHEST_COLLAPSE (Assuming a Closed Pneumothorax) """
        if injury_type == InjuryTypeEnum.CHEST_COLLAPSE:
            if injury_location != InjuryLocationEnum.LEFT_CHEST and \
                    injury_location != InjuryLocationEnum.LEFT_SIDE and \
                    injury_location != InjuryLocationEnum.RIGHT_CHEST and \
                    injury_location != InjuryLocationEnum.RIGHT_SIDE:
                issue = f"{injury_location} {injury_type} is not supported"
                if issue not in opts["issues"]:
                    opts["issues"].append(issue)
                continue

            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.DECOMPRESSION_NEEDLE]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.FENTANYL_LOLLIPOP]))
            continue

        """ EAR_BLEED """
        if injury_type == InjuryTypeEnum.EAR_BLEED:
            if injury_location != InjuryLocationEnum.HEAD and \
                    injury_location != InjuryLocationEnum.LEFT_FACE and \
                    injury_location != InjuryLocationEnum.RIGHT_FACE:
                issue = f"{injury_location} {injury_type} is not supported"
                if issue not in opts["issues"]:
                    opts["issues"].append(issue)
                continue

            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.FENTANYL_LOLLIPOP]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.HEMOSTATIC_GAUZE]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PRESSURE_BANDAGE]))
            continue

        """ INTERNAL """
        if injury_type == InjuryTypeEnum.INTERNAL:
            if injury_location != InjuryLocationEnum.INTERNAL:
                issue = f"{injury_location} {injury_type} is not supported"
                if issue not in opts["issues"]:
                    opts["issues"].append(issue)
                continue

            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.IV_BAG]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.BLOOD]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.FENTANYL_LOLLIPOP]))
            continue

        """ OPEN ABDOMINAL WOUND """
        if injury_type == InjuryTypeEnum.OPEN_ABDOMINAL_WOUND:
            if injury_location != InjuryLocationEnum.STOMACH and \
                    injury_location != InjuryLocationEnum.LEFT_STOMACH and \
                    injury_location != InjuryLocationEnum.RIGHT_STOMACH:
                issue = f"{injury_location} {injury_type} is not supported"
                if issue not in opts["issues"]:
                    opts["issues"].append(issue)
                continue

            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.HEMOSTATIC_GAUZE]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.IV_BAG]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.BLOOD]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.FENTANYL_LOLLIPOP]))
            continue

        """ TRAUMATIC BRAIN INJURY """
        if injury_type == InjuryTypeEnum.TRAUMATIC_BRAIN_INJURY:
            if injury_location != InjuryLocationEnum.HEAD:
                issue = f"{injury_location} {injury_type} is not supported"
                if issue not in opts["issues"]:
                    opts["issues"].append(issue)
                continue

            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.PAIN_MEDICATIONS]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.FENTANYL_LOLLIPOP]))
            recommend(_apply_treatment(injury_location, supplies,
                                       supply_precedence=[SupplyTypeEnum.IV_BAG]))
            continue

    return opts


"""
BELOW HERE IS DRIVER LOGIC FOR DEVELOPMENT AND TESTING
"""


def process_input_output(file: Path, treated_only: bool = True) -> dict:
    treatments = {}
    eval_input_output = None
    if file is not None and file.exists():
        _logger.info(f"Parsing {file}")
        with open(file, 'r') as stream:
            eval_input_output = json.load(stream)
    if not eval_input_output:
        _logger.error(f"Unable to parse file {file}")
        return treatments

    for scenario in eval_input_output:
        treated_character_id = ""
        action = scenario["output"]["action"]
        if action["action_type"] == "APPLY_TREATMENT":
            treated_character_id = action["character_id"]

        scenario_id = scenario["input"]["scenario_id"]
        state = scenario["input"]["full_state"]
        time = state["elapsed_time"]
        scene_id = state["meta_info"]["scene_id"]
        target = "no_tgt"
        if "alignment_target_id" in scenario["input"]:
            target = scenario["input"]["alignment_target_id"]
        _logger.info(f"{scenario_id}-{target}-{scene_id}-{time}")

        characters = state["characters"]
        for character in characters:

            character_treatment = {"treatment_applied": False}
            if treated_character_id and character["id"] == treated_character_id:
                character_treatment["treatment_applied"] = True
            if treated_only and not character_treatment["treatment_applied"]:
                continue

            idx = f"{scenario_id}-{target}-{scene_id}-{time}-{character['id']}"
            #character_treatment["recommended"] = recommend_treatments(character["injuries"], state["supplies"])
            character_treatment["options"] = treatment_options(character["injuries"], state["supplies"])
            character_treatment["supplies"] = state["supplies"]

            treatments[idx] = character_treatment

    return treatments


def process_scenario(file: Path):
    processed_injuries = {}
    scenario = None
    if file is not None and file.exists():
        _logger.info(f"Parsing {file}")
        with open(file, 'r') as stream:
            scenario = yaml.safe_load(stream)
    if not scenario:
        _logger.error(f"Unable to parse file {file}")
        return None

    def process_state(state):
        for character in state["characters"]:
            recommend_treatments(character["injuries"], state["supplies"])

    # Process the initial state
    process_state(scenario["state"])
    for scene in scenario["scenes"]:
        if "state" in scene:
            process_state(scene["state"])

    _logger.info("\nAll of the unique injuries and our suggested treatments for this json...")
    for desc, resps in processed_injuries.items():
        _logger.info(f"\t{desc}")
        for s in resps["suggestions"]:
            _logger.info(f"\t\t{s}")


def main():
    """Main function to run the client."""
    parser = argparse.ArgumentParser(description="Generate ITM YAML scenarios of our IRB study survey")
    parser.add_argument(
        "-r", "--results_dir",
        type=Path,
        default=None,
        help="Scenario to load"
    )
    args = parser.parse_args()
    folder = args.results_dir

    now = "_dev"  # datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s",
                        filename=f"competency_reporter_{now}.log", filemode="w")
    logging.getLogger("itm").setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    #process_input_output(Path("C:/Programming/ITM/dryrun-eval-results/competence_experiment_outputs/outlines_baseline_eval"))
    #process_input_output(Path("C:/Programming/ITM/dryrun-eval-results/live_results/comparative_regression_icl_template_eval_live/2024-08-23__20-00-38"))
    all_characters = process_input_output(Path("C:/Programming/ITM/dryrun-eval-results/live_results/group/cr/input_output.json"))
    #process_scenario(Path("C:/Programming/ITM/scenarios/qol-ph1-train-1-ta3.yaml"))

    prefix = '\t'
    unique_injury_treatments = {}
    unique_treatment_recommendations = set()
    for idx, character_treatments in all_characters.items():

        if "recommended" in character_treatments:
            treatable = False
            for recommended in character_treatments["recommended"]:
                if recommended["parameters"]:
                    treatable = True
                    if recommended["injury"] not in unique_injury_treatments:
                        unique_injury_treatments[recommended["injury"]] = []
                    option_treatments = unique_injury_treatments[recommended["injury"]]
                    if recommended["treatment"] not in option_treatments:
                        option_treatments.append(recommended["treatment"])
            if not treatable:
                _logger.error(f"{idx} is not treatable")
                if len(character_treatments["recommended"]) == 0:
                    _logger.error(f"\t...they don't have any injuries")
                if character_treatments["treatment_applied"]:
                    _logger.error(f"\t...but we tried to treat them anyway")
                for recommended in character_treatments["recommended"]:
                    _logger.error(f"\t{recommended['injury']}")
                _logger.error(f"{_supplies_to_str(character_treatments['supplies'], prefix) }")

        if "options" in character_treatments:
            if len(character_treatments["options"]["treatments"]) == 0:
                _logger.error(f"Character has no recommendations {idx}")
                if len(character_treatments["options"]["injuries"]) == 0:
                    _logger.error("\tCharacter has no injuries")
                else:
                    for injury in character_treatments["options"]["injuries"]:
                        _logger.error(f"\t{injury}")
                    _logger.error(f"{_supplies_to_str(character_treatments['supplies'], prefix) }")
            else:
                _logger.info(f"{idx}")
                for injury in character_treatments["options"]["injuries"]:
                    _logger.error(f"\t{injury}")
                for recommendation in character_treatments["options"]["treatments"]:
                    _logger.error(f"\t\t{recommendation}")
                    unique_treatment_recommendations.add(recommendation)
                _logger.error(f"{_supplies_to_str(character_treatments['supplies'], prefix) }")

    _logger.info("\nAll of the unique injuries and our suggested treatments for this json...")
    for injury, treatments in unique_injury_treatments.items():
        _logger.info(f"\t{injury}")
        for option in treatments:
            _logger.info(f"\t\t{option}")

    _logger.info("\nAll of the unique injury recommendations...")
    for recommendation in unique_treatment_recommendations:
        _logger.info(f"\t{recommendation}")


if __name__ == "__main__":
    main()
