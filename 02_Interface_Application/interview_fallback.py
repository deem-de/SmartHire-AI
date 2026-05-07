"""Offline interview generation and evaluation fallback.

This module keeps the demo genuinely useful when no API key is available.
It does not try to imitate a frontier model perfectly, but it should still:
- ask relevant questions from the parsed resume
- score answers differently based on real content
- vary strengths/improvements/suggested answers
- support both English and Arabic cleanly
"""

from __future__ import annotations

import re


def generate_template_questions(
    profile: dict,
    language: str = "English",
    number_of_questions: int = 3,
) -> list[dict]:
    """Generate practical interview questions without an external API."""

    skills = profile.get("Skills", [])
    projects = profile.get("Projects", [])
    experience = profile.get("Experience", []) or profile.get("Designation", [])

    questions: list[dict] = []

    if skills:
        skill = skills[0]
        questions.append(
            _question(
                english=f"How have you used {skill} in a real project?",
                arabic=f"كيف استخدمت {skill} في مشروع حقيقي؟",
                question_type="Technical",
                based_on=f"Skill: {skill}",
            )
        )

    if len(skills) > 1:
        skill = skills[1]
        questions.append(
            _question(
                english=f"What challenge did you face while working with {skill}?",
                arabic=f"ما التحدي الذي واجهته أثناء العمل على {skill}؟",
                question_type="Technical",
                based_on=f"Skill: {skill}",
            )
        )

    if projects:
        project = projects[0]
        questions.append(
            _question(
                english=f"Can you explain your project: {project}?",
                arabic=f"هل يمكنك شرح مشروعك: {project}؟",
                question_type="Project-Based",
                based_on=f"Project: {project}",
            )
        )

    if experience:
        exp = experience[0]
        questions.append(
            _question(
                english=f"What did you learn from your experience as {exp}?",
                arabic=f"ماذا تعلمت من خبرتك كـ {exp}؟",
                question_type="Experience",
                based_on=f"Experience: {exp}",
            )
        )

    questions.append(
        _question(
            english="Describe a difficult technical problem you solved and how you solved it.",
            arabic="صف مشكلة تقنية صعبة تعاملت معها وكيف قمت بحلها.",
            question_type="Behavioral",
            based_on="General interview assessment",
        )
    )

    selected = questions[:number_of_questions]
    if language == "Arabic":
        for item in selected:
            item["question"] = item.pop("arabic_question")
    else:
        for item in selected:
            item.pop("arabic_question")
    return selected


def evaluate_template_answer(answer: str, language: str = "English") -> dict:
    """Return varied, content-aware feedback without an external API."""

    words = answer.split()
    lower_answer = answer.lower()
    word_count = len(words)

    weak_phrases_en = [
        "i don't know",
        "i dont know",
        "don't remember",
        "dont remember",
        "just used it because",
        "copied",
        "everything good",
        "every thing good",
        "no idea",
    ]
    weak_phrases_ar = ["ما أعرف", "مدري", "ما أدري", "ما أتذكر", "نسيت", "ما عندي فكرة"]

    detail_keywords_en = [
        "project",
        "problem",
        "solution",
        "data",
        "model",
        "result",
        "because",
        "used",
        "built",
        "improved",
        "tested",
        "dashboard",
        "api",
        "deployment",
    ]
    detail_keywords_ar = [
        "مشروع",
        "مشكلة",
        "حل",
        "بيانات",
        "نموذج",
        "نتيجة",
        "استخدمت",
        "بنيت",
        "طورت",
        "حللت",
        "اختبرت",
        "واجهة",
        "تطبيق",
    ]

    has_weak_phrase = any(phrase in lower_answer for phrase in weak_phrases_en) or any(
        phrase in answer for phrase in weak_phrases_ar
    )
    detail_hits = sum(1 for token in detail_keywords_en if token in lower_answer) + sum(
        1 for token in detail_keywords_ar if token in answer
    )
    has_measurement = bool(re.search(r"\b\d+\b", answer)) or any(
        token in answer for token in ["%", "month", "months", "week", "weeks", "شهر", "أشهر", "أسبوع"]
    )
    has_example = any(
        token in lower_answer for token in ["for example", "for instance", "such as", "in my project", "for a client"]
    ) or any(token in answer for token in ["مثلاً", "على سبيل المثال", "في مشروعي", "في أحد المشاريع"])
    has_tool = any(
        token in lower_answer
        for token in [
            "python",
            "sql",
            "fastapi",
            "bert",
            "api",
            "dashboard",
            "excel",
            "power bi",
            "tableau",
            "tensorflow",
            "pytorch",
            "java",
            "c++",
        ]
    ) or any(token in answer for token in ["بايثون", "جافا", "فاستابي", "إكسل", "باور بي آي", "سي كيو إل", "واجهة برمجة"])
    has_result_word = any(
        token in lower_answer
        for token in ["result", "improved", "increase", "reduced", "achieved", "accuracy", "saved", "faster", "impact"]
    ) or any(token in answer for token in ["نتيجة", "تحسن", "زاد", "انخفض", "حقق", "سرعة", "دقة", "أثر"])
    has_challenge = any(
        token in lower_answer for token in ["challenge", "problem", "issue", "difficulty", "obstacle", "bug"]
    ) or any(token in answer for token in ["تحدي", "مشكلة", "صعوبة", "عقبة", "خطأ"])
    has_action = any(
        token in lower_answer
        for token in ["used", "built", "created", "implemented", "designed", "developed", "analyzed", "trained", "fixed"]
    ) or any(token in answer for token in ["استخدمت", "بنيت", "أنشأت", "نفذت", "صممت", "طورت", "حللت", "دربت", "أصلحت"])

    if has_weak_phrase or word_count < 6:
        score = 1
    elif word_count < 12:
        score = 2
    else:
        score = 2
        if detail_hits >= 2 or (has_tool and has_action):
            score += 1
        if has_example or has_challenge:
            score += 1
        if has_measurement or (has_result_word and detail_hits >= 3):
            score += 1

    score = max(1, min(score, 5))

    shared = {
        "score": score,
        "word_count": word_count,
        "detail_hits": detail_hits,
        "has_action": has_action,
        "has_tool": has_tool,
        "has_example": has_example,
        "has_result_word": has_result_word,
        "has_measurement": has_measurement,
        "has_challenge": has_challenge,
        "has_weak_phrase": has_weak_phrase,
    }

    if language == "Arabic":
        strengths = _arabic_strengths(**shared)
        improvements = _arabic_improvements(**shared)
        suggested = _arabic_suggested_answer(has_tool, has_challenge, has_measurement)
    else:
        strengths = _english_strengths(**shared)
        improvements = _english_improvements(**shared)
        suggested = _english_suggested_answer(has_tool, has_challenge, has_measurement)

    return {
        "score_out_of_5": score,
        "strength": " ".join(strengths[:2]),
        "improvement": " ".join(improvements[:2]),
        "suggested_better_answer": suggested,
    }


def create_template_summary(results: list[dict], language: str = "English") -> str:
    """Create a final summary when no external API is available."""

    scores = []
    for result in results:
        feedback = result.get("feedback", {})
        score = feedback.get("score_out_of_5", feedback.get("score", 0))
        if isinstance(score, (int, float)):
            scores.append(float(score))

    average = sum(scores) / len(scores) if scores else 0

    if language == "Arabic":
        recommendation = "أداء جيد للمقابلة الأولية" if average >= 3.5 else "يحتاج أمثلة أوضح وتفاصيل تقنية أقوى"
        return (
            f"الأداء العام: {average:.1f}/5\n"
            "نقاط القوة: حاول المرشح ربط الإجابات بخبرته ومهاراته العملية.\n"
            "نقاط التحسين: أضف أمثلة أوضح، والأدوات المستخدمة، ونتائج قابلة للقياس.\n"
            f"التوصية: {recommendation}."
        )

    recommendation = "Strong enough for an initial interview" if average >= 3.5 else "Needs clearer examples and stronger technical detail"
    return (
        f"Overall performance: {average:.1f}/5\n"
        "Strengths: The candidate attempted to connect answers to real experience.\n"
        "Areas to improve: Add clearer examples, concrete tools, and measurable results.\n"
        f"Recommendation: {recommendation}."
    )


def _english_strengths(
    *,
    score: int,
    word_count: int,
    detail_hits: int,
    has_action: bool,
    has_tool: bool,
    has_example: bool,
    has_result_word: bool,
    has_measurement: bool,
    has_challenge: bool,
    has_weak_phrase: bool,
) -> list[str]:
    strengths: list[str] = []

    if score <= 2:
        if has_tool:
            strengths.append("You named a concrete tool or technology, which gives the answer a real anchor.")
        if has_action:
            strengths.append("You hinted at what you personally did instead of staying completely abstract.")
        if word_count >= 8 and not has_weak_phrase:
            strengths.append("The answer has a usable base and can become stronger with one clear project example.")
    else:
        if score >= 4 and (has_result_word or has_measurement):
            strengths.append("You linked the answer to a clear outcome or measurable result, which makes the experience sound real.")
        if score >= 4 and has_challenge:
            strengths.append("You explained the challenge you faced, which makes the answer feel more mature and complete.")
        if has_action:
            strengths.append("You clearly described your own contribution instead of staying vague.")
        if has_tool:
            strengths.append("You mentioned specific tools or technologies, which makes the answer more credible.")
        if has_example:
            strengths.append("You included a concrete example, which helps the interviewer picture the work.")
        if has_challenge and score < 4:
            strengths.append("You acknowledged a challenge, which makes the story feel more realistic and complete.")
        if (has_result_word or has_measurement) and score < 4:
            strengths.append("You linked the answer to an outcome or measurable result, which is a strong interview signal.")
        if detail_hits >= 4:
            strengths.append("The answer carries enough technical detail to sound hands-on rather than memorized.")

    if not strengths:
        strengths.append("The answer addresses the topic, but it still needs clearer evidence of hands-on experience.")
    return strengths


def _english_improvements(
    *,
    score: int,
    word_count: int,
    detail_hits: int,
    has_action: bool,
    has_tool: bool,
    has_example: bool,
    has_result_word: bool,
    has_measurement: bool,
    has_challenge: bool,
    has_weak_phrase: bool,
) -> list[str]:
    improvements: list[str] = []

    if word_count < 12:
        improvements.append("Add a little more detail so the answer feels complete rather than rushed.")
    if not has_example:
        improvements.append("Add one clear example from a real project, internship, or task.")
    if not has_tool:
        improvements.append("Mention the exact tool, framework, or technology you used.")
    if not has_challenge:
        improvements.append("Explain one challenge you faced and how you handled it.")
    if not has_result_word and not has_measurement:
        improvements.append("End with a clear result or measurable impact.")
    if score >= 4 and has_example and has_tool and has_challenge and not has_measurement:
        improvements.insert(0, "This is already strong; the best upgrade would be adding one quantified result.")
    if score >= 4 and has_example and has_tool and (has_result_word or has_measurement):
        improvements.insert(0, "The answer is strong overall; refine it by making the story shorter and more specific.")

    if not improvements:
        improvements.append("The answer is already solid; tighten the structure and keep the best details.")
    return improvements


def _english_suggested_answer(has_tool: bool, has_challenge: bool, has_measurement: bool) -> str:
    parts = [
        "start with the problem or task",
        "explain what you personally did",
    ]
    if has_tool:
        parts.append("name the tools or technologies you used")
    else:
        parts.append("mention the exact tools or technologies you used")
    if not has_challenge:
        parts.append("describe one challenge you faced")
    if has_measurement:
        parts.append("finish with the final result")
    else:
        parts.append("finish with the final result and, if possible, one measurable outcome")
    return "A stronger answer should " + ", ".join(parts) + "."


def _arabic_strengths(
    *,
    score: int,
    word_count: int,
    detail_hits: int,
    has_action: bool,
    has_tool: bool,
    has_example: bool,
    has_result_word: bool,
    has_measurement: bool,
    has_challenge: bool,
    has_weak_phrase: bool,
) -> list[str]:
    strengths: list[str] = []

    if score <= 2:
        if has_tool:
            strengths.append("ذكرت أداة أو تقنية محددة، وهذه بداية جيدة تجعل الإجابة أكثر واقعية.")
        if has_action:
            strengths.append("وضحت جزءًا من دورك الشخصي بدل أن تبقى الإجابة عامة بالكامل.")
        if word_count >= 8 and not has_weak_phrase:
            strengths.append("الإجابة تعتبر بداية مقبولة ويمكن تقويتها بمثال أوضح من تجربة حقيقية.")
    else:
        if score >= 4 and (has_result_word or has_measurement):
            strengths.append("ربطت إجابتك بنتيجة أو أثر واضح، وهذا يعطي انطباعًا قويًا بأن التجربة حقيقية.")
        if score >= 4 and has_challenge:
            strengths.append("شرحت التحدي الذي واجهته، وهذا يجعل الإجابة أكثر نضجًا واكتمالًا.")
        if has_action:
            strengths.append("شرحت دورك الشخصي بوضوح، وهذا يعطي انطباعًا بأنك شاركت فعلاً في العمل.")
        if has_tool:
            strengths.append("ذكرت أدوات أو تقنيات محددة، وهذا يزيد مصداقية الإجابة.")
        if has_example:
            strengths.append("أضفت مثالًا عمليًا، وهذا يساعد على فهم خبرتك بشكل أوضح.")
        if has_challenge and score < 4:
            strengths.append("ذكرت التحدي الذي واجهته، وهذا يجعل الإجابة أكثر واقعية ونضجًا.")
        if (has_result_word or has_measurement) and score < 4:
            strengths.append("ربطت إجابتك بنتيجة أو أثر واضح، وهذه نقطة قوية جدًا في المقابلات.")
        if detail_hits >= 4:
            strengths.append("الإجابة فيها تفاصيل تقنية كافية وتبدو عملية أكثر من كونها محفوظة.")

    if not strengths:
        strengths.append("الإجابة بداية جيدة، لكنها ما زالت تحتاج أدلة أوضح على الخبرة العملية.")
    return strengths


def _arabic_improvements(
    *,
    score: int,
    word_count: int,
    detail_hits: int,
    has_action: bool,
    has_tool: bool,
    has_example: bool,
    has_result_word: bool,
    has_measurement: bool,
    has_challenge: bool,
    has_weak_phrase: bool,
) -> list[str]:
    improvements: list[str] = []

    if word_count < 12:
        improvements.append("أضف تفاصيل أكثر قليلًا حتى لا تبدو الإجابة مختصرة زيادة عن اللازم.")
    if not has_example:
        improvements.append("أضف مثالًا واضحًا من مشروع أو تدريب أو مهمة حقيقية.")
    if not has_tool:
        improvements.append("اذكر الأداة أو التقنية أو الإطار الذي استخدمته بشكل مباشر.")
    if not has_challenge:
        improvements.append("اشرح التحدي الذي واجهته وكيف تعاملت معه.")
    if not has_result_word and not has_measurement:
        improvements.append("اختم الإجابة بنتيجة واضحة أو أثر يمكن ملاحظته أو قياسه.")
    if score >= 4 and has_example and has_tool and has_challenge and not has_measurement:
        improvements.insert(0, "الإجابة قوية أصلًا، وما ينقصها غالبًا فقط رقم أو نتيجة قابلة للقياس.")
    if score >= 4 and has_example and has_tool and (has_result_word or has_measurement):
        improvements.insert(0, "الإجابة قوية جدًا، ويمكن تحسينها أكثر باختصار القصة وربطها مباشرة بالسؤال.")

    if not improvements:
        improvements.append("الإجابة جيدة بالفعل، فقط رتبها بشكل أوضح واحتفظ بأقوى التفاصيل.")
    return improvements


def _arabic_suggested_answer(has_tool: bool, has_challenge: bool, has_measurement: bool) -> str:
    parts = [
        "ابدأ بالمشكلة أو الهدف",
        "ثم اشرح ماذا فعلت أنت شخصيًا",
    ]
    if has_tool:
        parts.append("واذكر الأدوات التي استخدمتها")
    else:
        parts.append("واذكر الأداة أو التقنية المستخدمة")
    if not has_challenge:
        parts.append("وبيّن التحدي الذي واجهته")
    if has_measurement:
        parts.append("واختم بالنتيجة النهائية")
    else:
        parts.append("واختم بالنتيجة النهائية ويفضل أن تضيف رقمًا أو أثرًا واضحًا")
    return "إجابة أقوى ممكن تكون كذا: " + "، ".join(parts) + "."


def _question(
    english: str,
    arabic: str,
    question_type: str,
    based_on: str,
) -> dict:
    return {
        "question": english,
        "arabic_question": arabic,
        "type": question_type,
        "based_on": based_on,
        "difficulty": "Medium",
    }
