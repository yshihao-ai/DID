from __future__ import annotations

__all__ = [
    # High-level API
    "single_file",
    "single_file_structural",
    "single_file_rhythmic",
    "pair_eval",
    "pair_eval_structural",
    "distribution_eval",
    "advanced_eval",
    # Result containers
    "SingleFileResult",
    "RhythmicResult",
    "PairResult",
    "DistributionResult",
    "AdvancedResult",
    "StructuralSingleResult",
    "StructuralPairResult",
    # Individual metrics
    "compute_ca",
    "midi_to_chords",
    "midi_to_chords_dp",
    "recognize_chords",
    "recognize_chords_beat",
    "compute_cs",
    "extract_chord_vectors",
    "clear_cs_model_cache",
    "compute_ook",
    "compute_ook_percentage",
    "chord_histogram_entropy",
    "ngram_diversity",
    "melody_matchness",
    "tonal_distance",
    "mean_ioi",
    "rhythmic_intensity",
    "rhythmic_density",
    "voice_number",
    "onset_xor_distance",
    "note_overlap",
    "grooving_pattern_similarity",
    # Low-level modules
    "muspy_ext",
    "note_f1",
    "similarity",
    "chord_accuracy",
    "chord_recognition",
    "distribution",
    "advanced",
    "structural",
    "rhythmic",
]

# Lazy import map: name -> (module, attribute)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # High-level API
    "single_file": ("smg_metrics.single", "single_file"),
    "single_file_structural": ("smg_metrics.single", "single_file_structural"),
    "single_file_rhythmic": ("smg_metrics.single", "single_file_rhythmic"),
    "pair_eval": ("smg_metrics.pair", "pair_eval"),
    "pair_eval_structural": ("smg_metrics.pair", "pair_eval_structural"),
    "distribution_eval": ("smg_metrics.distribution", "compute_all"),
    "advanced_eval": ("smg_metrics.advanced", "compute_all"),
    # Result containers
    "SingleFileResult": ("smg_metrics.single", "SingleFileResult"),
    "RhythmicResult": ("smg_metrics.rhythmic", "RhythmicResult"),
    "PairResult": ("smg_metrics.pair", "PairResult"),
    "DistributionResult": ("smg_metrics.distribution", "DistributionResult"),
    "AdvancedResult": ("smg_metrics.advanced", "AdvancedResult"),
    "StructuralSingleResult": ("smg_metrics.structural", "StructuralSingleResult"),
    "StructuralPairResult": ("smg_metrics.structural", "StructuralPairResult"),
    # Individual metrics
    "compute_ca": ("smg_metrics.chord_accuracy", "compute_ca"),
    "midi_to_chords": ("smg_metrics.chord_accuracy", "midi_to_chords"),
    "midi_to_chords_dp": ("smg_metrics.chord_accuracy", "midi_to_chords_dp"),
    "recognize_chords": ("smg_metrics.chord_recognition", "recognize_chords"),
    "recognize_chords_beat": ("smg_metrics.chord_recognition", "recognize_chords_beat"),
    "compute_cs": ("smg_metrics.chord_similarity", "compute_cs"),
    "extract_chord_vectors": ("smg_metrics.chord_similarity", "extract_chord_vectors"),
    "clear_cs_model_cache": ("smg_metrics.chord_similarity", "clear_cs_model_cache"),
    "compute_ook": ("smg_metrics.out_of_key", "compute_ook"),
    "compute_ook_percentage": ("smg_metrics.out_of_key", "compute_ook_percentage"),
    "chord_histogram_entropy": ("smg_metrics.structural", "chord_histogram_entropy"),
    "ngram_diversity": ("smg_metrics.structural", "ngram_diversity"),
    "melody_matchness": ("smg_metrics.structural", "melody_matchness"),
    "tonal_distance": ("smg_metrics.structural", "tonal_distance"),
    "mean_ioi": ("smg_metrics.rhythmic", "mean_ioi"),
    "rhythmic_intensity": ("smg_metrics.rhythmic", "rhythmic_intensity"),
    "rhythmic_density": ("smg_metrics.rhythmic", "rhythmic_density"),
    "voice_number": ("smg_metrics.rhythmic", "voice_number"),
    "onset_xor_distance": ("smg_metrics.rhythmic", "onset_xor_distance"),
    "note_overlap": ("smg_metrics.rhythmic", "note_overlap"),
    "grooving_pattern_similarity": ("smg_metrics.rhythmic", "grooving_pattern_similarity"),
    # Low-level modules
    "muspy_ext": ("smg_metrics", "muspy_ext"),
    "note_f1": ("smg_metrics", "note_f1"),
    "similarity": ("smg_metrics", "similarity"),
    "chord_accuracy": ("smg_metrics", "chord_accuracy"),
    "chord_recognition": ("smg_metrics", "chord_recognition"),
    "distribution": ("smg_metrics", "distribution"),
    "advanced": ("smg_metrics", "advanced"),
    "structural": ("smg_metrics", "structural"),
    "rhythmic": ("smg_metrics", "rhythmic"),
}


def __getattr__(name: str):
    """Lazy import to speed up CLI startup."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value  # Cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return all public names for tab-completion and dir()."""
    return sorted(__all__)
