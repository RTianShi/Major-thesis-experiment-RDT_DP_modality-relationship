import re
import torch
from .registry import register_language


@register_language("identity")
def lang_identity(text_embed, cfg):
    return text_embed


@register_language("silence")
def lang_silence(text_embed, cfg):
    return torch.zeros_like(text_embed)


@register_language("empty_text")
def lang_empty_text(text_embed, cfg):
    # Use the empty-string embedding as language input.
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("empty_text requires cfg['encoder'] (text encoder) to be provided")
    empty = encoder([""])
    return empty


@register_language("gaussian_noise")
def lang_gaussian_noise(text_embed, cfg):
    sigma = float(cfg.get("sigma", 0.01))
    noise = torch.randn_like(text_embed) * sigma
    return text_embed + noise


@register_language("replace_red_with_blue")
def lang_replace_red_with_blue(text_embed, cfg):
    encoder = cfg.get("encoder")
    text = cfg.get("text")
    if encoder is None:
        raise ValueError("replace_red_with_blue requires cfg['encoder'] (text encoder) to be provided")
    if text is None:
        raise ValueError("replace_red_with_blue requires cfg['text'] (original instruction) to be provided")
    # Replace standalone 'red' (case-insensitive) with 'blue'
    mutated = re.sub(r"\bred\b", "blue", str(text), flags=re.IGNORECASE)
    return encoder([mutated])


@register_language("MR-JDCP-1")
def lang_jdcp1_equivalent_pick_transfer(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-JDCP-1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        "Pick up the red cube and transfer it to the destination.",
    )
    return encoder([str(mutated)])


@register_language("MR-JDCP-3")
def lang_jdcp3_irrelevant_or_empty(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-JDCP-3 requires cfg['encoder'] (text encoder) to be provided")
    if bool(cfg.get("use_empty", False)):
        mutated = ""
    else:
        mutated = cfg.get("mutated_text", "Tell me the color of the table.")
    return encoder([str(mutated)])


@register_language("MR-GDIP2")
def lang_global_language_deprivation(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-GDIP2 requires cfg['encoder'] (text encoder) to be provided")

    mode = str(cfg.get("mode", "do_nothing")).strip().lower()
    if mode == "empty":
        mutated = ""
    elif mode in {"do_nothing", "donothing", "noop"}:
        mutated = "do nothing"
    else:
        mutated = cfg.get("mutated_text", "")

    return encoder([str(mutated)])


@register_language("MR-JSAP-1")
def lang_replace_red_cube_with_blue_cylinder(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError(
            "MR-JSAP-1 requires cfg['encoder'] (text encoder) to be provided"
        )
    mutated = cfg.get(
        "mutated_text",
        "Grasp a blue cylinder and move it to a target goal position.",
    )
    return encoder([str(mutated)])


@register_language("MR-JSAP-2")
def lang_replace_blue_cylinder_with_blue_triangular_prism(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError(
            "MR-JSAP-2 requires cfg['encoder'] (text encoder) to be provided"
        )
    mutated = cfg.get(
        "mutated_text",
        "Grasp a blue triangular prism and move it to a target goal position.",
    )
    return encoder([str(mutated)])

@register_language("MR-JSAP-3")
def lang_replace_blue_cylinder_with_blue_triangular_prism(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError(
            "MR-JSAP-3 requires cfg['encoder'] (text encoder) to be provided"
        )
    mutated = cfg.get(
        "mutated_text",
        "Grasp a blue cube and move it to a target goal position.",
    )
    return encoder([str(mutated)])


@register_language("MR-DCRB1")
def lang_dcrb1_pick_blue_cube(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError(
            "MR-DCRB1 requires cfg['encoder'] (text encoder) to be provided"
        )
    mutated = cfg.get(
        "mutated_text",
        "Grasp a blue cube and move it to a target goal position.",
    )
    return encoder([str(mutated)])