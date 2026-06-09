"""
合规检测数据模型
==================
独立的数据类型定义，避免循环导入。
"""

import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Verdict(Enum):
    """检测结果判定"""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class RuleResult:
    """单项检测规则的结果"""
    rule_id: str
    category: str
    name: str
    verdict: Verdict
    score: int
    detail: str
    hint: Optional[str] = None
    critical: bool = False

    @property
    def is_pass(self) -> bool:
        return self.verdict == Verdict.PASS

    @property
    def is_fail(self) -> bool:
        return self.verdict == Verdict.FAIL

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "name": self.name,
            "verdict": self.verdict.value,
            "score": self.score,
            "detail": self.detail,
            "hint": self.hint,
            "critical": self.critical,
        }


@dataclass
class ComplianceReport:
    """合规检测报告"""
    standard: str
    rules: List[RuleResult] = field(default_factory=list)
    total_time: float = 0
    timestamp: str = ""

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.rules if r.is_pass)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.rules if r.is_fail)

    @property
    def warn_count(self) -> int:
        return sum(1 for r in self.rules if r.verdict == Verdict.WARN)

    @property
    def total_count(self) -> int:
        return len(self.rules)

    @property
    def overall_score(self) -> float:
        if not self.rules:
            return 0
        return float(np.mean([r.score for r in self.rules]))

    @property
    def is_compliant(self) -> bool:
        critical_fails = [r for r in self.rules if r.critical and r.is_fail]
        if critical_fails:
            return False
        non_critical_fails = [r for r in self.rules if not r.critical and r.is_fail]
        return len(non_critical_fails) <= 2

    @property
    def critical_failures(self) -> List[RuleResult]:
        return [r for r in self.rules if r.critical and r.is_fail]

    @property
    def by_category(self) -> Dict[str, List[RuleResult]]:
        cats = {}
        for r in self.rules:
            cats.setdefault(r.category, []).append(r)
        return cats

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  合规检测报告 -- 标准: {self.standard}",
            "=" * 60,
            f"  通过: {self.passed_count} | 失败: {self.failed_count} | "
            f"警告: {self.warn_count}",
            f"  综合评分: {self.overall_score:.0f}/100",
            f"  结果: {'[通过]' if self.is_compliant else '[未通过]'}",
            "",
        ]

        if self.critical_failures:
            lines.append("  [关键问题 -- 必须修复]")
            for r in self.critical_failures:
                lines.append(f"    {r.rule_id} {r.name}: {r.detail}")
                if r.hint:
                    lines.append(f"      -> {r.hint}")
            lines.append("")

        category_labels = {
            "geometric": "几何检测", "pose": "姿态检测",
            "facial": "面部状态", "lighting": "光照色彩",
            "quality": "图像质量",
        }
        for cat_key, cat_label in category_labels.items():
            cat_rules = self.by_category.get(cat_key, [])
            if not cat_rules:
                continue
            fails = [r for r in cat_rules if not r.is_pass]
            icon = "[OK]" if not fails else "[!!]"
            lines.append(f"  {icon} {cat_label}")

            for r in cat_rules:
                if r.is_pass and len(cat_rules) > 4:
                    continue
                v_icon = "PASS" if r.is_pass else ("WARN" if r.verdict == Verdict.WARN else "FAIL")
                lines.append(f"     {r.rule_id} [{v_icon:4s}] {r.name}: {r.detail}")
                if r.hint and not r.is_pass:
                    lines.append(f"            -> {r.hint}")

            lines.append("")

        lines.append(f"  总耗时: {self.total_time:.3f}s")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "standard": self.standard,
            "overall_score": round(self.overall_score, 1),
            "is_compliant": self.is_compliant,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "warn_count": self.warn_count,
            "total_count": self.total_count,
            "critical_failures": [
                {"rule_id": r.rule_id, "name": r.name, "hint": r.hint}
                for r in self.critical_failures
            ],
            "rules": [r.to_dict() for r in self.rules],
            "by_category": {
                cat: {"passed": sum(1 for r in rules if r.is_pass),
                      "failed": sum(1 for r in rules if r.is_fail)}
                for cat, rules in self.by_category.items()
            },
            "total_time": round(self.total_time, 3),
        }
