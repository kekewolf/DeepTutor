"""Guided Learning API Router."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from deeptutor.learning.models import ErrorType, QuizAttempt
from deeptutor.learning.scheduler import SpacedRepetitionScheduler
from deeptutor.learning.service import LearningService
from deeptutor.learning.storage import LearningStore

router = APIRouter()


def get_learning_service() -> LearningService:
    # Create a fresh store + service per request to avoid object-level race conditions.
    store = LearningStore()
    return LearningService(store)


def get_scheduler() -> SpacedRepetitionScheduler:
    # Stateless; safe to instantiate per request.
    return SpacedRepetitionScheduler()


# ── Request models ───────────────────────────────────────────────────────────


class AnswerRequest(BaseModel):
    question_id: str
    knowledge_point_id: str
    module_id: str = ""
    is_correct: bool
    user_answer: str | None = None
    error_type: str | None = None
    self_attribution: str = ""
    mastery_estimate: float = 0.0


class InitModulesRequest(BaseModel):
    modules: list[dict]  # list of LearningModule-compatible dicts


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/progress/{book_id}")
async def get_progress(book_id: str):
    service = get_learning_service()
    progress = service.get_or_create(book_id)
    return progress.model_dump()


@router.post("/progress/{book_id}/answer")
async def submit_answer(book_id: str, body: AnswerRequest):
    service = get_learning_service()
    scheduler = get_scheduler()

    progress = service.get_or_create(book_id)

    # 将字符串 error_type 转换为 ErrorType 枚举
    error_type_enum = None
    if body.error_type:
        try:
            error_type_enum = ErrorType(body.error_type)
        except ValueError:
            error_type_enum = None

    attempt = QuizAttempt(
        question_id=body.question_id,
        knowledge_point_id=body.knowledge_point_id,
        module_id=body.module_id,
        is_correct=body.is_correct,
        user_answer=body.user_answer,
        error_type=error_type_enum,
        self_attribution=body.self_attribution,
        mastery_estimate=body.mastery_estimate,
    )
    service.record_quiz_attempt(progress, attempt)

    # Update spaced repetition state
    kp_type = progress.knowledge_types.get(attempt.knowledge_point_id)
    if kp_type is not None:
        state = progress.repetition_states.get(attempt.knowledge_point_id)
        if state is None:
            # Auto-create initial repetition state for new knowledge points
            state = scheduler.get_initial_state(kp_type)
            progress.repetition_states[attempt.knowledge_point_id] = state
        scheduler.schedule_next(state, kp_type, attempt.is_correct)
        progress.review_queue = scheduler.build_review_queue(progress)

    # Update mastery estimate
    if attempt.mastery_estimate > 0:
        service.update_mastery(progress, attempt.knowledge_point_id, attempt.mastery_estimate)

    service.save(progress)
    return progress.model_dump()


@router.get("/progress/{book_id}/reviews")
async def get_reviews(book_id: str):
    service = get_learning_service()
    scheduler = get_scheduler()

    progress = service.get_or_create(book_id)
    tasks = scheduler.get_due_tasks(progress)
    return {"tasks": [t.model_dump() for t in tasks]}


@router.post("/progress/{book_id}/init-modules")
async def init_modules(book_id: str, body: InitModulesRequest):
    from pydantic import ValidationError as PydanticValidationError

    from deeptutor.learning.models import KnowledgePoint, LearningModule

    service = get_learning_service()
    progress = service.get_or_create(book_id)
    modules = []
    for i, m in enumerate(body.modules):
        kps_data = m.get("knowledge_points", [])
        try:
            kps = [KnowledgePoint(**kp) for kp in kps_data]
        except PydanticValidationError as exc:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=422,
                detail=f"Invalid knowledge_point data in modules[{i}]: {exc.errors()}",
            ) from exc
        # Remove knowledge_points from m to avoid duplicate argument to LearningModule
        m_clean = {k: v for k, v in m.items() if k != "knowledge_points"}
        try:
            modules.append(LearningModule(knowledge_points=kps, **m_clean))
        except PydanticValidationError as exc:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=422,
                detail=f"Invalid module data in modules[{i}]: {exc.errors()}",
            ) from exc
    service.init_modules(progress, modules)
    progress.current_module_id = modules[0].id if modules else ""
    progress.current_kp_index = 0
    service.save(progress)
    return {"status": "ok", "module_count": len(modules)}
