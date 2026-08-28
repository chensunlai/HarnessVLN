from schemas.navigation import JsonObject


NAV_STOP = "nav.stop"


def nav_stop_input_schema() -> JsonObject:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "minLength": 1},
            "reason": {"type": "string"},
            "actor": {"type": "string", "minLength": 1},
        },
        "required": ["status", "reason", "actor"],
        "additionalProperties": False,
    }


def nav_stop_output_schema() -> JsonObject:
    return nav_stop_input_schema()
