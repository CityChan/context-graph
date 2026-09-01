from pathlib import Path


GENERATOR = Path("scripts/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh")
SEEDS = Path("scripts/prepare_miroverse_full_policy_seeds_idev.sh")
SUBMIT = Path("scripts/submit_miroverse_full_policy_deepseek_v4_smoke.sh")
IDEV = Path("scripts/generate_miroverse_full_policy_deepseek_v4_5node_idev.sh")
IDEV4 = Path("scripts/generate_miroverse_full_policy_deepseek_v4_4node_idev.sh")


def test_generator_can_curate_complete_main_and_branch_policy():
    text = GENERATOR.read_text(encoding="utf-8")
    assert "FULL_POLICY_CURATOR=${FULL_POLICY_CURATOR:-0}" in text
    assert 'if [ "$FULL_POLICY_CURATOR" = "1" ]' in text
    assert "build_contextgraph_full_policy_sft.py" in text
    assert '--manifest "$SFT_MANIFEST"' in text


def test_seed_wrapper_uses_miroverse_musique_and_scratch():
    text = SEEDS.read_text(encoding="utf-8")
    assert "$SCRATCH/datasets/MiroVerse-v0.1/jsonl_sft/MiroVerse-MuSiQue.jsonl" in text
    assert "$SCRATCH/contextgraph_sft/miroverse_full_policy" in text
    assert "prepare_miroverse_search_policy_seeds.py" in text
    assert "prepare_miroverse_retrieval_corpus.py" in text
    assert "embed_local_search_corpus.py" in text
    assert "EMBED_ATTN_IMPLEMENTATION=${EMBED_ATTN_IMPLEMENTATION:-sdpa}" in text
    assert '--attn-implementation "$EMBED_ATTN_IMPLEMENTATION"' in text


def test_smoke_submitter_enables_complete_policy_curator():
    text = SUBMIT.read_text(encoding="utf-8")
    assert "MAX_SAMPLES=${MAX_SAMPLES:-20}" in text
    assert "FULL_POLICY_CURATOR=1" in text
    assert "generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh" in text


def test_idev_smoke_uses_one_search_and_four_teacher_nodes():
    text = IDEV.read_text(encoding="utf-8")
    assert "EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-5}" in text
    assert "TEACHER_TP=${TEACHER_TP:-4}" in text
    assert "MAX_SAMPLES=${MAX_SAMPLES:-20}" in text
    assert "FULL_POLICY_CURATOR=1" in text
    assert "generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh" in text
    assert "LOCAL_SEARCH_CORPUS" in text
    assert "LOCAL_SEARCH_EMBEDDINGS" in text


def test_four_node_idev_smoke_colocates_retrieval_conservatively():
    text = IDEV4.read_text(encoding="utf-8")
    assert "EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}" in text
    assert "TEACHER_TP=${TEACHER_TP:-4}" in text
    assert "COLOCATE_SEARCH=1" in text
    assert "MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}" in text
    assert "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.70}" in text
    assert "FULL_POLICY_CURATOR=1" in text
    assert "LOCAL_SEARCH_CORPUS" in text
    assert "LOCAL_SEARCH_EMBEDDINGS" in text
    assert 'test -s "$LOCAL_SEARCH_CORPUS"' in text
    assert 'test -s "$LOCAL_SEARCH_EMBEDDINGS"' in text


def test_generator_supports_explicit_colocated_search_layout():
    text = GENERATOR.read_text(encoding="utf-8")
    assert "COLOCATE_SEARCH=${COLOCATE_SEARCH:-0}" in text
    assert 'if [ "$COLOCATE_SEARCH" = "1" ]' in text
    assert "MODEL_NODE_START=0" in text
    assert 'colocated TEACHER_TP=$TEACHER_TP must equal node count $NUM_NODES' in text


def test_generator_supports_a_local_search_corpus_pair():
    text = GENERATOR.read_text(encoding="utf-8")
    assert "LOCAL_SEARCH_CORPUS and LOCAL_SEARCH_EMBEDDINGS must be set together" in text
    assert "--local-corpus $LOCAL_SEARCH_CORPUS" in text
    assert "--local-embeddings $LOCAL_SEARCH_EMBEDDINGS" in text
