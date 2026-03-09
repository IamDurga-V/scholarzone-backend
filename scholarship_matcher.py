import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


class ScholarshipMatcher:
    def __init__(self, db):
        self.db     = db
        self.model  = SentenceTransformer("all-MiniLM-L6-v2")
        self.index  = None
        self.stored = []

    def _profile_to_text(self, profile: dict) -> str:
        return (
            f"Student category {profile.get('category', 'general')} "
            f"from {profile.get('state', 'India')} "
            f"studying {profile.get('educationLevel', 'ug')} "
            f"{profile.get('course', '')} "
            f"with {profile.get('percentage', '60')} percent marks "
            f"annual family income {profile.get('annualIncome', '500000')} rupees "
            f"gender {profile.get('gender', 'female')} "
            f"disability {profile.get('disability', 'no')}"
        )

    def _scholarship_to_text(self, s: dict) -> str:
        elig = s.get("eligibility", {})
        return (
            f"Scholarship {s.get('name', '')} "
            f"for {' '.join(elig.get('categories', ['all']))} category students "
            f"from {' '.join(elig.get('states', ['all']))} "
            f"studying {' '.join(elig.get('educationLevels', ['all']))} "
            f"maximum income {elig.get('maxIncome', 0)} rupees "
            f"minimum marks {elig.get('minPercentage', 0)} percent "
            f"gender {elig.get('gender', 'all')} "
            f"amount {s.get('amount', 0)} rupees "
            f"{s.get('description', '')}"
        )

    def _hard_filter(self, profile: dict, scholarships: List[dict]) -> List[dict]:
        passed = []
        cat    = profile.get("category", "general")
        state  = profile.get("state", "")
        gender = profile.get("gender", "female")

        try:
            income = float(profile.get("annualIncome", 999999))
        except:
            income = 999999

        try:
            pct = float(profile.get("percentage", 0))
        except:
            pct = 0

        for s in scholarships:
            elig = s.get("eligibility", {})

            max_income = elig.get("maxIncome", 0)
            if max_income > 0 and income > max_income:
                continue

            min_pct = elig.get("minPercentage", 0)
            if min_pct > 0 and pct < min_pct:
                continue

            cats = elig.get("categories", ["all"])
            if "all" not in cats and cat not in cats:
                continue

            states = elig.get("states", ["all"])
            if "all" not in states and state not in states:
                continue

            req_gender = elig.get("gender", "all")
            if req_gender != "all" and req_gender != gender:
                continue

            passed.append(s)

        return passed

    def build_index(self, scholarships: List[dict]):
        if not scholarships:
            return
        texts      = [self._scholarship_to_text(s) for s in scholarships]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        dim        = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.stored = scholarships
        print(f"FAISS index built with {len(scholarships)} scholarships")

    def match(self, profile: dict, scholarships: List[dict], top_k: int = 20) -> List[dict]:
        filtered = self._hard_filter(profile, scholarships)
        print(f"Hard filter: {len(scholarships)} to {len(filtered)} scholarships")

        if not filtered:
            return []

        self.build_index(filtered)

        if self.index is None or self.index.ntotal == 0:
            return filtered[:top_k]

        profile_text = self._profile_to_text(profile)
        profile_vec  = self.model.encode([profile_text], show_progress_bar=False)
        profile_vec  = np.array(profile_vec).astype("float32")
        faiss.normalize_L2(profile_vec)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(profile_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            s = self.stored[idx].copy()
            s["faiss_score"] = float(score)
            results.append(s)

        return results