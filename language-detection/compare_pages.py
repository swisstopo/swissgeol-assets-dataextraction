import pandas as pd

df_new = pd.read_csv("data/pages_bu_20240207.csv", low_memory=False).set_index(["filename", "page_number"])
df_old = pd.read_csv("data/pages_old.csv").set_index(["filename", "page_number"])

combined = df_new.join(df_old, how="outer", lsuffix="_new", rsuffix="_old")

mismatch = combined[(abs(combined['height_new'] - combined['height_old']) > 0.01) | (abs(combined['width_new'] - combined['width_old']) > 0.01)]
print("{} pages with mismatched dimensions.".format(len(mismatch)))
print()
mismatch.to_csv("data/mismatch_dimensions.csv")

only_new = combined[combined["height_old"].isna()]
print("{} pages that are only in the new file.".format(len(only_new)))
print()
only_new.to_csv("data/mismatch_only_new.csv")

only_old = combined[combined["height_new"].isna()]
print("{} pages that are only in the old file.".format(len(only_old)))
print()
only_old.to_csv("data/mismatch_only_old.csv")

combined['text_change'] = combined['ocr_type_old'] + '->' + combined['ocr_type_new']
combined['language_change'] = combined['language_code_old'].fillna("") + '->' + combined['language_code_new'].fillna("")

print(combined['ocr_type_old'].value_counts())
print()
print(combined['ocr_type_new'].value_counts())
print()
print(combined['text_change'].value_counts())
print()
print(combined['language_change'].value_counts().head(20))
print()
