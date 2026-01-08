import argostranslate.translate as tr

langs = tr.get_installed_languages()
codes = [l.code for l in langs]
print("Installed languages:", codes)

pairs = set()
for src in langs:
    # Newer versions often use translations_to
    if hasattr(src, "translations_to"):
        for dst in langs:
            try:
                t = src.get_translation(dst)
                if t is not None:
                    pairs.add((src.code, dst.code))
            except Exception:
                pass
    else:
        # Older versions: src.translations exists
        for t in getattr(src, "translations", []):
            pairs.add((t.from_lang.code, t.to_lang.code))

print("Installed translation pairs:", sorted(pairs))
