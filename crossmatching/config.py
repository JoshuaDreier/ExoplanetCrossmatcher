import configparser
from pathlib import Path

_cfg = configparser.ConfigParser()
_cfg.read_dict({
    "id_supplier": {"input_col": "input_ids", "id_col": "id", "null_sentinel": "--"},
    "crossmatcher": {
        "match_type_key": "match_type",
        "id_match_label": "id",
        "coord_match_label": "coordinates",
        "angular_sep_key": "angular_separation",
    },
})
_cfg.read(Path(__file__).parent / "crossmatching.cfg")

id_supplier = _cfg["id_supplier"]
crossmatcher = _cfg["crossmatcher"]
