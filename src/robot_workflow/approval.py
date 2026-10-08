"""One-time approval tokens cryptographically bound to immutable plans."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import secrets
import time

from .domain import MotionPlan
from .errors import ApprovalError


@dataclass(frozen=True)
class ApprovalToken:
    plan_id: str
    value: str


@dataclass
class _ApprovalRecord:
    plan_id: str
    plan_digest: str
    scene_id: str
    reviewer: str
    token_hash: str
    approved_at_s: float
    expires_at_s: float
    consumed: bool = False


class ApprovalAuthority:
    """In-memory authority for a single orchestrator process.

    Production persistence can replace this class, but it must preserve the
    exact digest, expiry, one-time use, and audit semantics.
    """

    def __init__(self) -> None:
        self._records: dict[str, _ApprovalRecord] = {}

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def approve(
        self,
        plan: MotionPlan,
        *,
        presented_digest: str,
        reviewer: str,
        now_s: float | None = None,
    ) -> ApprovalToken:
        """Record an explicit human decision after the preview was presented."""

        now = time.time() if now_s is None else now_s
        if not reviewer.strip():
            raise ApprovalError("reviewer identity is required")
        if now >= plan.expires_at_s:
            raise ApprovalError("cannot approve an expired plan")
        if not hmac.compare_digest(presented_digest, plan.digest):
            raise ApprovalError("presented plan digest does not match the immutable plan")
        existing = self._records.get(plan.plan_id)
        if existing:
            raise ApprovalError("plan was already approved; create a new plan to review again")

        raw_token = secrets.token_urlsafe(32)
        self._records[plan.plan_id] = _ApprovalRecord(
            plan_id=plan.plan_id,
            plan_digest=plan.digest,
            scene_id=plan.scene_id,
            reviewer=reviewer,
            token_hash=self._token_hash(raw_token),
            approved_at_s=now,
            expires_at_s=plan.expires_at_s,
        )
        return ApprovalToken(plan_id=plan.plan_id, value=raw_token)

    def consume(
        self,
        plan: MotionPlan,
        token: ApprovalToken,
        *,
        now_s: float | None = None,
    ) -> None:
        """Consume an approval immediately before execution."""

        now = time.time() if now_s is None else now_s
        if token.plan_id != plan.plan_id:
            raise ApprovalError("approval token belongs to a different plan")
        record = self._records.get(plan.plan_id)
        if record is None:
            raise ApprovalError("plan has not been approved")
        if record.consumed:
            raise ApprovalError("approval token has already been consumed")
        if now >= min(record.expires_at_s, plan.expires_at_s):
            raise ApprovalError("approval has expired")
        if not hmac.compare_digest(record.plan_digest, plan.digest):
            raise ApprovalError("plan content changed after approval")
        if record.scene_id != plan.scene_id:
            raise ApprovalError("plan scene changed after approval")
        if not hmac.compare_digest(record.token_hash, self._token_hash(token.value)):
            raise ApprovalError("invalid approval token")
        record.consumed = True

    def is_consumed(self, plan_id: str) -> bool:
        record = self._records.get(plan_id)
        return bool(record and record.consumed)
