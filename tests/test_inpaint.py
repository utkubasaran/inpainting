from inpaint import build_inpaint_prompt


def test_build_inpaint_prompt_mentions_selected_part_and_description():
    prompt = build_inpaint_prompt("top", "white blouse")

    assert "white blouse" in prompt
    assert "upper body garment" in prompt


def test_build_inpaint_prompt_trims_whitespace():
    prompt = build_inpaint_prompt("shoes", "  navy heels  ")

    assert "navy heels" in prompt
    assert "  navy heels  " not in prompt
