"""StackCube-specific language mutations."""

from .registry import register_language


MR1_STACKCUBE_DEFAULT_MUTATED_TEXT = (
    "Grasp the red block and place it onto the green box, ensuring it stays."
)
MR5_STACKCUBE_DEFAULT_MUTATED_TEXT = (
    "Grasp the red cube and stack it vertically on the green cube."
)


@register_language("MR-SADP1")
@register_language("Semantic Agnostic Decisiveness")
def lang_mr_sadp1_semantic_agnostic_decisiveness(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-SADP1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        (
            "Locate the crimson block, elevate it securely, and position it directly atop "
            "the verdant cube. Finally, release your grip to ensure it rests statically."
        ),
    )
    return encoder([str(mutated)])


@register_language("MR-E1")
@register_language("DCRB-Role-Reversal")
def lang_mr_e1_role_reversal(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-E1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        "Pick up the green cube and stack it on top of the red cube.",
    )
    return encoder([str(mutated)])


@register_language("MR-B1")
@register_language("JSAP-CoMutate-Color")
def lang_mr_b1_color_comutate(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-B1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        "Pick up a blue cube and stack it on top of a yellow cube.",
    )
    return encoder([str(mutated)])



@register_language("MR1")
@register_language("MR-1")
@register_language("Synonym-Substitution")
def lang_mr1_synonym_substitution(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        MR1_STACKCUBE_DEFAULT_MUTATED_TEXT,
    )
    return encoder([str(mutated)])


@register_language("MR-A1")
@register_language("JDCP-Equivalent-Lang")
def lang_mr_a1_equivalent_lang(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-A1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        MR1_STACKCUBE_DEFAULT_MUTATED_TEXT,
    )
    return encoder([str(mutated)])


@register_language("MR5")
@register_language("MR-5")
@register_language("Instruction-Specialization")
def lang_mr5_instruction_specialization(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR5 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        MR5_STACKCUBE_DEFAULT_MUTATED_TEXT,
    )
    return encoder([str(mutated)])
