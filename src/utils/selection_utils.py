#selection_utils.py

def update_selection(selected_keys, edited_df):

    current_selected = set(
        edited_df[edited_df["Select"] == True].index
    )

    added = current_selected - selected_keys
    removed = selected_keys - current_selected

    return current_selected, added, removed