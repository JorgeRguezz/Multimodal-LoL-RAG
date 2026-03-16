from knowledge_inference.answer_postprocess import inject_video_urls, prettify_video_name


def test_prettify_video_name_replaces_underscores():
    raw = "The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends"
    pretty = prettify_video_name(raw)
    assert pretty == "The COMPLETE SMOLDER STARTER GUIDE - League of Legends"


def test_inject_video_urls_replaces_exact_video_name_in_answer():
    answer = "See The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends for the main example."
    registry = {
        "The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends": "https://www.youtube.com/watch?v=abc123"
    }

    updated = inject_video_urls(answer, registry)

    assert (
        updated
        == "See The COMPLETE SMOLDER STARTER GUIDE - League of Legends (https://www.youtube.com/watch?v=abc123) for the main example."
    )


def test_inject_video_urls_leaves_unknown_names_unchanged():
    answer = "See unknown_video for the main example."
    registry = {
        "The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends": "https://www.youtube.com/watch?v=abc123"
    }

    updated = inject_video_urls(answer, registry)

    assert updated == answer
