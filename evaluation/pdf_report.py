from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .models import EvaluationReport


_CORE_METRICS = (
    ("sentiment_accuracy", "Sentiment"),
    ("intent_accuracy", "Intent"),
    ("category_accuracy", "Category"),
    ("priority_accuracy", "Priority"),
    ("action_selection_accuracy", "Action selection"),
    ("conditional_route_accuracy", "Conditional routing"),
)
_ALL_METRICS = _CORE_METRICS + (
    ("node_execution_accuracy", "Graph node execution"),
    ("tool_selection_accuracy", "Tool selection (operational-node proxy)"),
    ("tool_execution_accuracy", "Tool execution (operational-node proxy)"),
)
_JUDGE_METRICS = (
    ("relevance", "Relevance"),
    ("helpfulness", "Helpfulness"),
    ("empathy", "Empathy"),
    ("accuracy", "Response accuracy"),
    ("clarity", "Clarity"),
)


def _text(value: Any) -> str:
    if value is None:
        return "Not applicable"
    if isinstance(value, bool):
        return "Allowed" if value else "Rejected"
    if isinstance(value, list):
        return ", ".join(_text(item) for item in value) if value else "None"
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_text(item)}" for key, item in value.items())
    return str(value).replace("_", " ")


def _paragraph(value: Any, style: ParagraphStyle) -> Paragraph:
    safe = escape(_text(value)).replace("\n", "<br/>")
    return Paragraph(safe, style)


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _table(
    rows: list[list[Any]],
    widths: list[float],
    *,
    header: bool = True,
    font_size: int = 8,
) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [
            colors.white,
            colors.HexColor("#F1F5F9"),
        ]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def _bar_chart(
    labels: list[str],
    values: list[float],
    width: float,
) -> Drawing | None:
    if not labels:
        return None
    chart_height = max(120, 25 * len(labels))
    drawing = Drawing(width, chart_height + 25)
    chart = HorizontalBarChart()
    chart.x = 150
    chart.y = 12
    chart.width = max(250, width - 165)
    chart.height = chart_height
    chart.data = [[value * 100 for value in values]]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.dx = -3
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 100
    chart.valueAxis.valueStep = 20
    chart.valueAxis.labelTextFormat = "%d%%"
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = colors.HexColor("#2B7A78")
    chart.bars[0].strokeColor = colors.HexColor("#1F5F5D")
    drawing.add(chart)
    return drawing


def _execution_pie(summary: dict[str, Any]) -> Drawing | None:
    execution = summary["executions"]
    values = [
        execution["completed"],
        execution["guardrail_rejected"],
        execution["failed"],
    ]
    if not sum(values):
        return None
    labels = [
        f"Completed ({values[0]})",
        f"Guardrail rejected ({values[1]})",
        f"Execution failed ({values[2]})",
    ]
    drawing = Drawing(460, 180)
    chart = Pie()
    chart.x = 150
    chart.y = 12
    chart.width = 155
    chart.height = 155
    chart.data = values
    chart.labels = labels
    chart.simpleLabels = 1
    chart.slices.strokeColor = colors.white
    chart.slices.strokeWidth = 1
    slice_colors = (
        colors.HexColor("#2B7A78"),
        colors.HexColor("#E3A638"),
        colors.HexColor("#C44536"),
    )
    for index, color in enumerate(slice_colors):
        chart.slices[index].fillColor = color
    drawing.add(chart)
    return drawing


def _case_explanation(result: dict[str, Any]) -> str:
    status = result["execution_status"]
    if status == "failed":
        return (
            "The case could not be completed. Its execution failure is recorded "
            f"separately from prediction scores: {_text(result.get('execution_error'))}"
        )
    if status == "guardrail_rejected":
        return (
            "A guardrail rejected the input, so the graph was not run and prediction "
            "metrics were not scored."
        )
    if not result.get("prediction_errors"):
        return "Pass: all applicable expected labels and graph actions matched."

    expected = result["expected"]
    predicted = result.get("predicted") or {}
    problems: list[str] = []
    labels = {
        "sentiment_accuracy": "sentiment",
        "intent_accuracy": "intent",
        "category_accuracy": "category",
        "priority_accuracy": "priority",
    }
    for metric, field in labels.items():
        if metric in result["prediction_errors"]:
            problems.append(
                f"{field.title()} expected '{_text(expected.get(field))}' but observed "
                f"'{_text(predicted.get(field))}'"
            )
    if "action_selection_accuracy" in result["prediction_errors"]:
        problems.append(
            "Expected actions: "
            f"{_text(expected.get('actions'))}; observed actions: "
            f"{_text(predicted.get('actions'))}"
        )
    if "conditional_route_accuracy" in result["prediction_errors"]:
        problems.append("The graph took a different conditional route than expected.")
    if problems:
        return "Needs review: " + ". ".join(problems) + "."
    return "Some graph-node or operational-action checks did not match."


def _case_table(result: dict[str, Any], styles: Any, width: float) -> Table:
    expected = result["expected"]
    predicted = result.get("predicted") or {}
    rows = [
        ["Field", "Expected", "Observed"],
        ["Sentiment", expected.get("sentiment"), predicted.get("sentiment")],
        ["Category", expected.get("category"), predicted.get("category")],
        ["Intent", expected.get("intent"), predicted.get("intent")],
        ["Priority", expected.get("priority"), predicted.get("priority")],
        ["Actions", expected.get("actions"), predicted.get("actions")],
        ["Graph nodes", "Expected action path", result.get("nodes", [])],
        ["Guardrails", "Custom and LLM allow", result.get("guardrails", {})],
        [
            "Simulated effects",
            "No real delivery or persistence",
            f"{result.get('simulated_tickets', 0)} ticket(s); "
            f"{result.get('simulated_emails', 0)} simulated email(s)",
        ],
    ]
    formatted = [
        [
            _paragraph(cell, styles["table_cell"])
            for cell in row
        ]
        for row in rows
    ]
    if result.get("judge_scores"):
        judge_text = ", ".join(
            f"{key.title()}: {value}/5"
            for key, value in result["judge_scores"].items()
        )
        formatted.append(
            [
                _paragraph("LLM judge", styles["table_cell"]),
                _paragraph("Not applicable", styles["table_cell"]),
                _paragraph(judge_text, styles["table_cell"]),
            ]
        )
    if result.get("judge_error"):
        formatted.append(
            [
                _paragraph("LLM judge error", styles["table_cell"]),
                _paragraph("Judge score expected", styles["table_cell"]),
                _paragraph(result["judge_error"], styles["table_cell"]),
            ]
        )
    if result.get("execution_error"):
        formatted.append(
            [
                _paragraph("Execution error", styles["table_cell"]),
                _paragraph("Successful graph execution", styles["table_cell"]),
                _paragraph(result["execution_error"], styles["table_cell"]),
            ]
        )

    return _table(
        formatted,
        [width * 0.20, width * 0.37, width * 0.43],
        font_size=8,
    )


def write_pdf_report(
    report: EvaluationReport,
    guardrails: dict[str, Any],
    output_path: Path,
    *,
    regression: dict[str, Any] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Customer Feedback Agent Evaluation",
        author="Local Evaluation Framework",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=25,
            textColor=colors.HexColor("#17324D"),
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=12,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CaseHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#245B72"),
            spaceBefore=10,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="table_cell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            wordWrap="CJK",
        )
    )

    def page_footer(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.line(16 * mm, 12 * mm, A4[0] - 16 * mm, 12 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(16 * mm, 8 * mm, "Customer Feedback Agent Evaluation")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    summary = report.summary
    width = doc.width
    story: list[Any] = [
        Paragraph("Customer Feedback Agent Evaluation", styles["ReportTitle"]),
        _paragraph(
            f"Agent version: {report.agent_version} | "
            f"Dataset: {Path(report.dataset).name} | "
            f"Started: {report.started_at} | Finished: {report.finished_at}",
            styles["ReportBody"],
        ),
        _paragraph(
            "This report evaluates synthetic cases. Email delivery and database "
            "writes were simulated; no customer email was sent.",
            styles["ReportBody"],
        ),
        Paragraph("Overall performance", styles["SectionHeading"]),
    ]

    overall = summary["overall_performance"]
    execution = summary["executions"]
    headline_rows = [
        ["Measure", "Result", "How to read it"],
        [
            "Total average performance",
            _percent(overall["overall_average_accuracy"]),
            f"Macro-average accuracy across {overall['averaged_metrics']} applicable core metrics",
        ],
        [
            "Full-case pass rate",
            _percent(overall["case_pass_rate"]),
            f"{overall['fully_correct_cases']} of {overall['evaluated_cases']} completed cases passed every applicable core check",
        ],
        [
            "Graph executions",
            f"{execution['completed']}/{execution['total']} completed",
            f"{execution['failed']} failed; {execution['guardrail_rejected']} rejected by guardrails",
        ],
        [
            "LLM judge average",
            (
                f"{summary['llm_judge']['overall_mean']:.2f}/5"
                if summary["llm_judge"]["overall_mean"] is not None
                else "Not run"
            ),
            "Separate model-based ratings of drafted responses",
        ],
    ]
    story.append(
        _table(
            [
                [_paragraph(cell, styles["table_cell"]) for cell in row]
                for row in headline_rows
            ],
            [width * 0.25, width * 0.20, width * 0.55],
        )
    )
    story.append(Spacer(1, 7))
    story.append(
        _paragraph(
            "Overall average is the unweighted mean of sentiment, intent, category, "
            "priority, action-selection, and conditional-route accuracy. A metric "
            "marked not applicable is excluded. Full-case pass rate is stricter: "
            "every applicable core metric for that case must be correct.",
            styles["ReportSmall"],
        )
    )

    metric_rows = [["Metric", "Accuracy", "Evaluated cases"]]
    for key, label in _ALL_METRICS:
        value = summary[key]
        metric_rows.append(
            [
                _paragraph(label, styles["table_cell"]),
                _percent(value["accuracy"]),
                str(value["evaluated"]),
            ]
        )
    story.extend(
        [
            Paragraph("Metric averages", styles["SectionHeading"]),
            _table(
                metric_rows,
                [width * 0.58, width * 0.21, width * 0.21],
            ),
            Spacer(1, 8),
        ]
    )

    chart_labels: list[str] = []
    chart_values: list[float] = []
    for key, label in _CORE_METRICS:
        value = summary[key]["accuracy"]
        if value is not None:
            chart_labels.append(label)
            chart_values.append(value)
    performance_chart = _bar_chart(
        chart_labels,
        chart_values,
        width,
    )
    if performance_chart is not None:
        story.extend(
            [
                Paragraph("Core metric chart", styles["SectionHeading"]),
                performance_chart,
            ]
        )

    execution_chart = _execution_pie(summary)
    if execution_chart is not None:
        story.extend(
            [
                Paragraph("Execution outcomes", styles["SectionHeading"]),
                execution_chart,
            ]
        )

    judge_rows = [["Judge dimension", "Average score", "Scale"]]
    for key, label in _JUDGE_METRICS:
        dimension = summary["llm_judge"][key]
        mean = dimension["mean"]
        judge_rows.append(
            [
                label,
                f"{mean:.2f}" if mean is not None else "Not run",
                "1–5",
            ]
        )
    story.extend(
        [
            Paragraph("LLM-as-a-Judge results", styles["SectionHeading"]),
            _table(
                [
                    [_paragraph(cell, styles["table_cell"]) for cell in row]
                    for row in judge_rows
                ],
                [width * 0.50, width * 0.25, width * 0.25],
            ),
        ]
    )

    guardrail_rows = [
        [
            "Guardrail",
            "True accepts",
            "True rejects",
            "False accepts",
            "False rejects",
            "Accuracy",
        ]
    ]
    for key, label in (
        ("custom_guardrail", "Custom"),
        ("llm_guardrail", "LLM"),
    ):
        values = guardrails[key]
        guardrail_rows.append(
            [
                label,
                str(values["true_acceptance"]),
                str(values["true_rejection"]),
                str(values["false_acceptance"]),
                str(values["false_rejection"]),
                _percent(values["accuracy"]),
            ]
        )
    story.extend(
        [
            Paragraph("Guardrail results", styles["SectionHeading"]),
            _table(
                [
                    [_paragraph(cell, styles["table_cell"]) for cell in row]
                    for row in guardrail_rows
                ],
                [width * value for value in (0.18, 0.15, 0.15, 0.17, 0.17, 0.18)],
                font_size=7,
            ),
            _paragraph(
                f"Guardrail evaluation errors: custom "
                f"{guardrails['failed_custom_checks']}; LLM "
                f"{guardrails['failed_llm_checks']}.",
                styles["ReportSmall"],
            ),
        ]
    )
    guardrail_case_rows = [
        [
            "Guardrail case",
            "Custom expected",
            "Custom observed",
            "LLM expected",
            "LLM observed",
            "Explanation",
        ]
    ]
    for case in guardrails["cases"]:
        has_error = case["custom_error"] or case["llm_error"]
        matches = (
            case["custom_expected"] == case["custom_actual"]
            and case["llm_expected"] == case["llm_actual"]
        )
        if has_error:
            explanation = (
                f"Check error: {case['custom_error'] or ''} "
                f"{case['llm_error'] or ''}"
            )
        elif matches:
            explanation = "Both guardrails matched their expected decisions."
        else:
            explanation = (
                "Review this decision: one or both observed results differ "
                "from the expected labels."
            )
        guardrail_case_rows.append(
            [
                case["case_id"],
                _text(case["custom_expected"]),
                _text(case["custom_actual"]),
                _text(case["llm_expected"]),
                _text(case["llm_actual"]),
                explanation,
            ]
        )
    if len(guardrail_case_rows) == 1:
        guardrail_case_rows.append(["No guardrail cases were evaluated.", "", "", "", "", ""])
    story.extend(
        [
            Paragraph("Guardrail case-by-case explanations", styles["CaseHeading"]),
            _table(
                [
                    [_paragraph(cell, styles["table_cell"]) for cell in row]
                    for row in guardrail_case_rows
                ],
                [
                    width * part
                    for part in (0.22, 0.13, 0.13, 0.13, 0.13, 0.26)
                ],
                font_size=7,
            ),
        ]
    )

    if regression is not None:
        regression_rows = [["Metric", "Baseline", "Candidate", "Change"]]
        for metric, value in regression["metrics"].items():
            regression_rows.append(
                [
                    metric.replace("_", " ").title(),
                    _percent(value["baseline"]),
                    _percent(value["candidate"]),
                    _percent(value["delta"]),
                ]
            )
        story.extend(
            [
                Paragraph("Regression comparison", styles["SectionHeading"]),
                _paragraph(
                    f"Baseline: {regression['baseline_version']} | "
                    f"Candidate: {regression['candidate_version']}",
                    styles["ReportBody"],
                ),
                _table(
                    [
                        [_paragraph(cell, styles["table_cell"]) for cell in row]
                        for row in regression_rows
                    ],
                    [width * 0.40, width * 0.20, width * 0.20, width * 0.20],
                ),
            ]
        )

    story.extend(
        [
            PageBreak(),
            Paragraph("Case-by-case explanations", styles["SectionHeading"]),
            _paragraph(
                "Expected and observed values are shown side by side. A failed "
                "execution is distinguished from a completed case with an incorrect "
                "prediction. Raw customer email addresses are omitted.",
                styles["ReportBody"],
            ),
        ]
    )
    for index, result in enumerate(report.results, start=1):
        title = Paragraph(
            f"Case {index}: {escape(result.case_id)} "
            f"({escape(result.execution_status.replace('_', ' '))})",
            styles["CaseHeading"],
        )
        explanation = _paragraph(_case_explanation(result.model_dump()), styles["ReportBody"])
        story.append(KeepTogether([title, explanation]))
        story.append(_case_table(result.model_dump(), styles, width))
        story.append(Spacer(1, 10))

    story.extend(
        [
            Paragraph("How to interpret tool metrics", styles["SectionHeading"]),
            _paragraph(
                "The production graph currently contains no LangChain tool nodes. "
                "Tool-selection and tool-execution values are proxies based on "
                "support-ticket and internal-notification graph actions; do not "
                "interpret them as independent tool-call metrics.",
                styles["ReportSmall"],
            ),
        ]
    )
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return output_path
