"""Recognize the reviewed path-only migration of the published scoring pipeline.

The archived environments and checkpoint signatures remain byte-exact. Fresh
runs still record their actual source hashes. Only the listed source fields can
match a migrated digest; model, data and preprocessing must still match exactly.
"""
MIGRATED_TO_ARCHIVED = {
    # P1/prompts/common.py
    "b39fd5d51fb5ae29cdc3a262aed380a154270a5cd66d2ab455501e944d7cfe41": "8a0b34c28f9487715f2971840ef7f6a764d209433e4a6fd78a6f3bd14fcbc380",
    # P1/prompts/score_new.py
    "2bd051b19776e272eff981b1c09826d1aa71a8fc3fafc4f466d663940f028e99": "f35498e71332d1eabc571d0c574411c04e9c48e27c315300c2a89169ca97174f",
    # P2/study.py
    "8db2ea5ac6be4362007c313ab3b41e2f9691df0cc64719ea2639b57aff22841c": "99e626c518d5e36cc59f42264b4ff06e6e63a8011655af1201775acf12c9ce62",
    # P2/score_p2.py
    "0180bb748a6f6f804c95bf4a74e5c1319818b045c7bc16de864d0d48bc173389": "44ca9d2d070c6f63cb5ec16f28eec084db045a74f899aab7801dc364840338a5",
    # P3/common.py
    "bd94981e294aecb2ba35120deaa23a54d2a2c1b1153684aa50b50065a3a78a0f": "55a0f9f43ca51da4eff4fbe8b118d39c17844e3936520730bf3ea9f1eba0b8cf",
    # P3/score_p3.py
    "2fe519e66a43c1a09598bb5c2b0fca2c29aff70a190f7152da1c656a1390c547": "113c30c7de9743d0618ed4002df86824b4735ef4c806b3d50e9078ca5c66f7f3",
}


def matches_digest(recorded, current):
    return current == recorded or MIGRATED_TO_ARCHIVED.get(current, current) == recorded


def matches_environment(recorded, current, source_fields):
    if recorded.keys() != current.keys():
        return False
    for key, value in current.items():
        saved = recorded[key]
        if key not in source_fields:
            if saved != value:
                return False
        elif isinstance(value, dict):
            if value.keys() != saved.keys() or not all(matches_digest(saved[k], v) for k, v in value.items()):
                return False
        elif not matches_digest(saved, value):
            return False
    return True
