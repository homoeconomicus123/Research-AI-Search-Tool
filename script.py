import requests
import csv
from openai import OpenAI
import ast
import re
import os

client = OpenAI(api_key="")


def extract_keywords_with_gpt(title, abstract):
    prompt = f"""
    You are a technical assistant that extracts structured, detailed insights from research paper titles and abstracts.

    Your primary focus is to identify research on additive manufacturing of monocrystalline (single-crystal) materials. Contextual relevance to “additive manufacturing” and “monocrystalline/single crystal” is the highest priority—only then consider experimental details, materials, methods, equipment, and so on.

    Given the following title and abstract, extract the following fields in Python dictionary format. Use **single quotes** for all keys and string values. Your answers should be in **full sentences** or **technical phrases**, resembling those found in research summaries.

    If a field is not clearly mentioned or inferable from the content, return:
    - An empty list for list-type fields
    - An empty string for string-type fields
    - Zero for the `relevance_score` if no abstract is provided

    Extract:
    1. 'method' (list): Scientific or engineering methods or processes used. Example: ['Selective Laser Melting (SLM)', 'Directed Energy Deposition (DED)']
    2. 'material' (list): Specific monocrystalline materials or alloys mentioned. Example: ['Ni-based superalloy single crystal', 'MgO single crystal']
    3. 'equipment' (list): Machines, tools, or specialized hardware used. Example: ['fiber laser system', 'crystal growth furnace']
    4. 'problem' (string): What scientific or technical issue the work addresses, phrased like a technical summary. Example: 'The paper addresses the challenge of controlling crystal orientation and defect density during laser-based fabrication of single-crystal turbine blades.'
    5. 'solution' (string): A technically phrased description of the proposed method, structure, or innovation. Example: 'The solution applies a tailored laser scan strategy combined with thermal gradient control to promote epitaxial growth and minimize grain boundaries in monocrystalline builds.'
    6. 'opinion' (string): Provide a concise technical opinion on the work, including its strengths, contributions, relevance, or implications. Use a neutral but informed tone, as if summarizing it for an engineering report or literature review. Example: 'This work contributes a novel thermal management approach that could significantly improve crystal quality and orientation control in single-crystal AM processes.'
    7. 'relevance_score' (integer): A numeric score from 0–5 indicating how directly the abstract focuses on additive manufacturing of monocrystalline materials, with this precise breakdown:
       - 5: Full experimental study explicitly on monocrystalline AM—clear additive manufacturing context, single-crystal focus, and detailed materials, methods, equipment, and results.
       - 4: Strong monocrystalline AM focus with experimental procedures and at least two of the following: detailed materials, detailed methods, or detailed equipment.
       - 3: Moderate monocrystalline AM relevance—experimental work is present but missing key details in materials, methods, or equipment.
       - 2: Weak monocrystalline or AM mention—some experimental context but neither monocrystalline nor AM is central.
       - 1: Only mentions additive manufacturing or monocrystalline in passing, with no experimental detail.
       - 0: No abstract provided or no mention of monocrystalline or additive manufacturing.
    8. 'relevance_reason' (string): A concise technical explanation of why the above score was assigned, referencing which aspects of the abstract (e.g., presence or absence of experiment details, monocrystalline focus, additive manufacturing context) determined the score.

    Only return a raw Python dictionary, without any markdown formatting, code blocks, or extra text. Do not include triple backticks or specify a language like `python`. Just return the dictionary itself.

    Title: {title}
    Abstract: {abstract}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful scientific extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        content = response.choices[0].message.content
        cleaned = content.strip().lstrip()
        cleaned = cleaned.replace('“', "'").replace('”', "'").replace('‘', "'").replace('’', "'")
        cleaned = re.sub(r'[^\x00-\x7F]+', '', cleaned)  # Remove non-ASCII
        parsed = ast.literal_eval(cleaned)

        return {
            "method": parsed.get("method", []),
            "material": parsed.get("material", []),
            "equipment": parsed.get("equipment", []),
            "problem": parsed.get("problem", ""),
            "solution": parsed.get("solution", ""),
            "opinion": parsed.get("opinion", ""),
            "relevance_score": parsed.get("relevance_score", ""),
            "relevance_reason": parsed.get("relevance_reason", "")
        }
    except Exception as e:
        print(f"❌ Failed to parse OpenAI response: {e}")
        return {
            "method": [],
            "material": [],
            "equipment": [],
            "problem": "",
            "solution": "",
            "opinion": "",
            "relevance_score": "",
            "relevance_reason":"",
        }


def read_work_ids_from_csv(input_file):
    work_ids = []
    with open(input_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            work_id = row.get("work_id")
            if work_id:
                work_ids.append(work_id.strip())
    return work_ids


def reconstruct_abstract(abstract_index):
    if not abstract_index:
        return None
    abstract_len = max(i for indices in abstract_index.values() for i in indices) + 1
    abstract_words = [""] * abstract_len
    for word, indices in abstract_index.items():
        for idx in indices:
            abstract_words[idx] = word
    return " ".join(abstract_words)


def fetch_openalex_data(work_ids):
    results = []

    for i, raw_id in enumerate(work_ids, start=1):
        print(f"✅ Processing {i} of {len(work_ids)}: {raw_id}")
        try:
            # Normalize in case some rows have the full URL
            work_id = raw_id.replace("https://openalex.org/", "").strip()
            url = f"https://api.openalex.org/works/{work_id}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # Basic fields with safe defaults
            title    = data.get("title", "")
            abstract = reconstruct_abstract(data.get("abstract_inverted_index", {})) or ""

            primary_location = data.get("primary_location") or {}
            source           = primary_location.get("source") or {}
            journal          = source.get("display_name", "")
            landing_url      = primary_location.get("landing_page_url", "")

            year     = data.get("publication_year", "")
            cited_by = data.get("cited_by_count", 0)

            # Authors & institutions
            authorships  = data.get("authorships", [])
            authors      = [a.get("author", {}).get("display_name", "") for a in authorships]
            institutions = []
            for a in authorships:
                for inst in a.get("institutions", []):
                    name = inst.get("display_name")
                    if name and name not in institutions:
                        institutions.append(name)

            # GPT extraction (including relevance_score & relevance_reason)
            gpt = extract_keywords_with_gpt(title, abstract)

            results.append({
                "work_id": work_id,
                "landing_url": landing_url,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "authors": ", ".join(authors),
                "institutions": ", ".join(institutions),
                "year": year,
                "cited_by": cited_by,
                "method": gpt.get("method", []),
                "material": gpt.get("material", []),
                "equipment": gpt.get("equipment", []),
                "problem": gpt.get("problem", ""),
                "solution": gpt.get("solution", ""),
                "opinion": gpt.get("opinion", ""),
                "relevance_score": gpt.get("relevance_score", 0),
                "relevance_reason": gpt.get("relevance_reason", "")
            })

        except Exception as e:
            print(f"⚠️ Skipping {raw_id}: {e}")
            results.append({
                "work_id": raw_id,
                "landing_url": "",
                "title": "",
                "abstract": "",
                "journal": "",
                "authors": "",
                "institutions": "",
                "year": "",
                "cited_by": "",
                "method": [],
                "material": [],
                "equipment": [],
                "problem": "",
                "solution": "",
                "opinion": "",
                "relevance_score": 0,
                "relevance_reason": ""
            })
            continue

    return results


def export_to_csv(data, output_file="openalex_results.csv"):
    keys = ["work_id", "landing_url", "title", "abstract", "journal", "authors", "institutions", "year", "cited_by",
            "method", "material", "equipment", "problem", "solution", "opinion", "relevance_score", "relevance_reason"]
    with open(output_file, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


input_csv = ""

work_ids = read_work_ids_from_csv(input_csv)[:700]
data = fetch_openalex_data(work_ids)
desktop = os.path.expanduser("~/Desktop")
output_csv = os.path.join(desktop, "output_v1.1.csv")
export_to_csv(data, output_csv)
print(f"✅ Done! {len(data)} records saved to '{output_csv}'")
