import anybadge
import coverage

cov = coverage.Coverage(config_file=".coveragerc")

cov.load()

coverage_report = cov.report()
coverage_percent_str = coverage_report.split()[1].strip("%")

coverage_percent = float(coverage_percent_str)

thresholds = {30: "red", 50: "orange", 70: "yellow", 80: "green", 100:"green"}

badge_value = f"{coverage_percent:.1f}%"
for threshold, color in thresholds.items():
    if coverage_percent >= threshold:
        badge_color = color
    else:
        break

badge_text = f"Coverage: {badge_value}"
badge = anybadge.Badge("Coverage", badge_text, badge_color)

badge.write_badge("coverage_badge.svg")
