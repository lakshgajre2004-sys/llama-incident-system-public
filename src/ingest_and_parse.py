# src/ingest_and_parse.py
import os, csv, re

RAW_PATH = "data/raw/hdfs.log"
OUT_DIR = "data/processed"

# known level tokens
LEVELS = {"INFO","WARN","WARNING","ERROR","DEBUG","FATAL","TRACE"}

def find_level_token(tokens):
    """Return (idx, level_str) if any level token found, else (None, None)."""
    for i, t in enumerate(tokens):
        # remove trailing punctuation
        tt = t.strip()
        if tt in LEVELS:
            return i, tt
    return None, None

def parse_line(line):
    # normalize spaces
    line = line.strip()
    if not line:
        return None
    tokens = line.split()
    idx, level = find_level_token(tokens)
    if idx is None:
        # fallback: try regex (e.g., 'INFO' inside)
        m = re.search(r"\b(INFO|WARN|WARNING|ERROR|DEBUG|FATAL|TRACE)\b", line)
        if m:
            level = m.group(1)
            # everything before match -> timestamp+maybe extra; after -> component/message
            before, after = line.split(level, 1)
            # try to split before into timestamp tokens
            before_tokens = before.strip().split()
            timestamp = " ".join(before_tokens[:2]) if len(before_tokens)>=2 else before.strip()
            # component = first word of after
            after = after.strip()
            if ":" in after:
                comp, msg = after.split(":",1)
            else:
                parts = after.split(None,1)
                comp = parts[0] if parts else ""
                msg = parts[1] if len(parts)>1 else ""
            return timestamp.strip(), level, comp.strip(), msg.strip()
        # no level found — return raw as message
        return "", "", "", line

    # build timestamp from tokens before level occurrence
    # assume timestamp is first two tokens (date/time) or first token if compact
    # join tokens[0:idx_of_level - maybe extra]
    # common pattern: tokens[0]=date, tokens[1]=time, tokens[idx] = level
    if idx >= 2:
        timestamp = tokens[0] + " " + tokens[1]
        # component is tokens between level and colon or first token after level
        after = tokens[idx+1:]
    else:
        timestamp = tokens[0]
        after = tokens[idx+1:]

    # try extract component and message from the remainder (after)
    remainder = " ".join(after)
    if ":" in remainder:
        comp, msg = remainder.split(":", 1)
    else:
        parts = remainder.split(None, 1)
        comp = parts[0] if parts else ""
        msg = parts[1] if len(parts)>1 else ""
    return timestamp.strip(), level, comp.strip(), msg.strip()

def parse_hdfs():
    os.makedirs(OUT_DIR, exist_ok=True)
    logs_out = os.path.join(OUT_DIR, "logs.csv")
    with open(RAW_PATH, "r", errors="ignore") as infile, \
         open(logs_out, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["timestamp","level","component","message"])
        for line in infile:
            parsed = parse_line(line)
            if parsed is None:
                continue
            timestamp, level, comp, msg = parsed
            writer.writerow([timestamp, level, comp, msg])
    print(f"[✔] Parsed logs saved to: {logs_out}")

if __name__ == "__main__":
    parse_hdfs()
