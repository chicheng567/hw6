"""
Kaggle 競賽指標：ROUGE-L F1-Score
---------------------------------
此指標用於自動評估文本摘要任務。它計算提交摘要 (candidate) 與標準答案摘要 (reference)
之間最長共同子序列 (Longest Common Subsequence, LCS) 的 F1-score。
"""

import pandas as pd
import pandas.api.types as ptypes

class ParticipantVisibleError(Exception):
    """讓參賽者可見的錯誤訊息必須透過此 Exception 丟出。"""
    pass


def _lcs_length(tokens_a: list[str], tokens_b: list[str]) -> int:
    """
    以標準動態規畫計算兩個 token 序列的 LCS 長度。
    為了省記憶體，僅保存前一列與當前列 (O(min(m,n)))。
    """
    if not tokens_a or not tokens_b:
        return 0

    # 確保第二個序列較短，節省記憶體
    if len(tokens_a) < len(tokens_b):
        tokens_a, tokens_b = tokens_b, tokens_a

    prev = [0] * (len(tokens_b) + 1)
    curr = [0] * (len(tokens_b) + 1)

    for token_a in tokens_a:
        for j, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, prev  # 交換陣列，避免重新配置
    return prev[-1]


def rouge_l_f1(reference: str, candidate: str) -> float:
    """
    計算單一 reference 與 candidate 的 ROUGE-L F1-score。
    """
    ref_tokens = reference.split()
    cand_tokens = candidate.split()

    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    if ref_len == 0 or cand_len == 0:
        return 0.0

    lcs = _lcs_length(ref_tokens, cand_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / cand_len
    recall = lcs / ref_len
    return 2 * precision * recall / (precision + recall)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    計算整份 submission 的平均 ROUGE-L F1-score。
    Kaggle 會保證 solution 與 submission 在 row_id 對齊。
    """

    # 刪去 Kaggle 已比對的 id 欄
    if row_id_column_name in solution.columns:
        del solution[row_id_column_name]
    if row_id_column_name in submission.columns:
        del submission[row_id_column_name]

    # --- Submission 驗證 ---
    if "summary" not in submission.columns:
        raise ParticipantVisibleError("Submission must have a 'summary' column.")
    if "summary" not in solution.columns:
        raise ValueError("Solution must have a 'summary' column. (Host Error)")

    if not (ptypes.is_string_dtype(submission["summary"]) or
            ptypes.is_object_dtype(submission["summary"])):
        raise ParticipantVisibleError('Submission column "summary" must be text.')
    if submission["summary"].isnull().any():
        raise ParticipantVisibleError("Submission 'summary' column contains missing values (NaN).")
    if solution["summary"].isnull().any():
        raise ValueError("Solution 'summary' column contains missing values. (Host Error)")

    references = solution["summary"].astype(str)
    candidates = submission["summary"].astype(str)

    scores = [
        rouge_l_f1(ref, cand)
        for ref, cand in zip(references, candidates)
    ]
    return float(sum(scores) / len(scores)) if scores else 0.0
