"""Définition des 12 classes FLAIR-INC et leurs couleurs RGB."""

INPUT_SIZE = 512

FLAIR_CLASS_NAMES: dict[int, str] = {
    0:  "building",
    1:  "pervious_surface",
    2:  "impervious_surface",
    3:  "bare_soil",
    4:  "water",
    5:  "coniferous",
    6:  "deciduous",
    7:  "brushwood",
    8:  "vineyard",
    9:  "herbaceous",
    10: "agricultural_land",
    11: "plowed_land",
}

FLAIR_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0:  (180, 110,  60),
    1:  (190, 190, 130),
    2:  (110, 110, 110),
    3:  (139,  90,  43),
    4:  ( 30, 144, 255),
    5:  (  0,  80,   0),
    6:  ( 34, 139,  34),
    7:  (107, 142,  35),
    8:  (128,   0, 128),
    9:  (124, 205, 124),
    10: (255, 215,   0),
    11: (160,  82,  45),
}
