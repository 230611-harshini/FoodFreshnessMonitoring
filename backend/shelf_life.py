SHELF_LIFE = {
    "fresh_bread": "3-5 days",
    "fresh_dairy": "5-7 days",
    "fresh_fruits": "5-10 days",
    "fresh_vegetables": "4-7 days",
    "spoiled_bread": "Expired",
    "spoiled_dairy": "Expired",
    "spoiled_fruits": "Expired",
    "spoiled_vegetables": "Expired"
}

def get_shelf_life(prediction):
    return SHELF_LIFE.get(prediction, "Unknown")