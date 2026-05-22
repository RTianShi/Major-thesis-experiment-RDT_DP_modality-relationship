"""PushCube language mutations.

Most generic language MRs live in `eval_sim.mr.language`. PushCube overrides
the equivalent-language MR locally because the task is push-based rather than
pick-and-place based.
"""

from eval_sim.mr.language import *  # noqa: F401,F403

from .registry import register_language


@register_language("MR1")
@register_language("MR-1")
@register_language("Synonym-Substitution")
def lang_mr1_synonym_substitution(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        "Nudge the block until it reaches the target circular area.",
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
        "Carefully push the cube in a straight line to the goal.",
    )
    return encoder([str(mutated)])


@register_language("MR-JDCP1")
@register_language("MR-JDCP-1")
@register_language("JDCP-Equivalent-Lang")
def lang_mr_jdcp1_equivalent_language(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-JDCP1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        "Slide the block forward until it reaches the target zone.",
    )
    return encoder([str(mutated)])


@register_language("MR-JDCP2")
@register_language("MR-JDCP-2")
@register_language("JDCP-Deprivative-Lang")
def lang_mr_jdcp2_deprivative_language(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-JDCP2 requires cfg['encoder'] (text encoder) to be provided")

    mode = str(cfg.get("mode", "empty")).strip().lower()
    if mode == "empty":
        mutated = ""
    elif mode in {"irrelevant", "conflict", "garbage"}:
        mutated = cfg.get("mutated_text", "Translate the book into French.")
    else:
        mutated = cfg.get("mutated_text", "")

    return encoder([str(mutated)])


@register_language("MR-JSAP1")
@register_language("MR-JSAP-1")
@register_language("JSAP-CoMutate-Goal")
def lang_mr_jsap1_goal_left_language(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-JSAP1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        "Push and move a cube to a goal region to its left.",
    )
    return encoder([str(mutated)])


@register_language("MR-SDPP1")
@register_language("MR-SDPP-1")
@register_language("SDPP-Spatial-Conflict")
def lang_mr_sdpp1_spatial_conflict(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-SDPP1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        "Push and move a cube to a goal region to its backward (behind it).",
    )
    return encoder([str(mutated)])


@register_language("MR-CMSI1")
@register_language("MR-CMSI-1")
@register_language("CMSI-Contrastive-Decoy-Targets")
def lang_mr_cmsi1_color_anchored_goal(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-CMSI1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        "Push and move a cube to the red/white goal region in front of it.",
    )
    return encoder([str(mutated)])


@register_language("MR-DCRB1")
@register_language("MR-DCRB-1")
@register_language("DCRB-OOD-Entity-Rebinding")
def lang_mr_dcrb1_ood_entity_rebinding(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-DCRB1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        "Push and move the yellow star-prism to a goal region in front of it.",
    )
    return encoder([str(mutated)])


@register_language("MR-DCRB2")
@register_language("MR-DCRB-2")
@register_language("DCRB-OOD-Object-Rebinding")
def lang_mr_dcrb2_ood_object_rebinding(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-DCRB2 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        "Push and move the yellow cube to a goal region in front of it.",
    )
    return encoder([str(mutated)])


@register_language("MR-SESP2")
@register_language("MR-SESP-2")
@register_language("SESP-Physical-Semantic-Synergy")
def lang_mr_sesp2_physical_semantic_synergy(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-SESP2 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        "Push the large heavy cube to the red white target.",
    )
    return encoder([str(mutated)])


@register_language("MR-SCDP2")
@register_language("MR-SCDP-2")
@register_language("SCDP-Linguistic-Syntax-Debunking")
def lang_mr_scdp2_linguistic_syntax_debunking(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-SCDP2 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        (
            "Hello robot arm, could you please carefully push that little block "
            "forward into the red and white circular zone, making sure it stays "
            "on the table? Thanks."
        ),
    )
    return encoder([str(mutated)])


@register_language("MR-SADP1")
@register_language("MR-SADP-1")
@register_language("SADP-Cross-Modal-Semantic-Noise")
def lang_mr_sadp1_cross_modal_semantic_noise(text_embed, cfg):
    encoder = cfg.get("encoder")
    if encoder is None:
        raise ValueError("MR-SADP1 requires cfg['encoder'] (text encoder) to be provided")
    mutated = cfg.get(
        "mutated_text",
        (
            "Gently slide the cubic object forward until it rests entirely "
            "within the designated circular target zone ahead of it."
        ),
    )
    return encoder([str(mutated)])
