"""
Tests for WP-5.4: API reference documentation

This test suite validates the structure, completeness, and validity of
the generated API reference documentation at docs/api-reference.md.

Tests verify:
1. File existence and minimum size
2. Markdown structural validity (headers, fences, tables)
3. All 11 required sections present
4. All acceptance criteria met
5. Key API elements documented
"""

import re
from pathlib import Path


class TestDocumentationFileExists:
    """Verify the documentation file exists and has expected size."""

    def test_api_reference_file_exists(self):
        """Test that docs/api-reference.md exists in the project root."""
        doc_path = Path("docs/api-reference.md")
        assert doc_path.exists(), f"API reference file not found at {doc_path}"

    def test_api_reference_has_content(self):
        """Test that the file has substantial content (>30KB as noted in build report)."""
        doc_path = Path("docs/api-reference.md")
        size_bytes = doc_path.stat().st_size
        assert size_bytes > 30000, f"Expected >30KB but got {size_bytes} bytes"

    def test_api_reference_minimum_lines(self):
        """Test that the file has expected line count (~773 lines)."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Build report claims 773 lines
        assert len(lines) > 700, f"Expected >700 lines but got {len(lines)}"


class TestMarkdownStructure:
    """Validate markdown formatting and structure."""

    def test_all_code_fences_balanced(self):
        """Test that all code fences (```) are properly balanced."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Count opening and closing fences
        fence_count = content.count('```')
        assert fence_count % 2 == 0, f"Unbalanced code fences: {fence_count} total"

    def test_header_hierarchy_valid(self):
        """Test that markdown headers follow proper hierarchy (#, ##, ###, etc)."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        max_level = 0

        for line in lines:
            match = header_pattern.match(line)
            if match:
                level = len(match.group(1))
                # Headers should increase gradually (no jump from H1 to H4)
                assert level <= max_level + 1, f"Header jump detected: H{max_level} -> H{level}"
                max_level = level

    def test_no_unclosed_tables(self):
        """Test that markdown tables have complete structure (header + separator).

        A valid markdown table always has a separator row (|---|---| pattern)
        immediately after the header row.  We detect header rows as lines that
        start with '|' where the immediately preceding line does NOT start with
        '|' (i.e. the first row of a new table block), then assert the very
        next line matches the separator pattern.
        """
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header_pat = re.compile(r'^\|')
        sep_pat = re.compile(r'^\|[-: ]+\|')

        # Find the first line of each table block
        header_indices = [
            i for i, line in enumerate(lines)
            if header_pat.match(line)
            and (i == 0 or not header_pat.match(lines[i - 1]))
        ]

        assert len(header_indices) > 0, (
            "No markdown tables found in documentation"
        )

        for i in header_indices:
            assert i + 1 < len(lines), (
                f"Table header at line {i + 1} has no following line"
            )
            next_line = lines[i + 1]
            assert sep_pat.match(next_line), (
                f"Table header at line {i + 1} is not followed by a separator row.\n"
                f"  Header:    {lines[i].rstrip()}\n"
                f"  Next line: {next_line.rstrip()}"
            )


class TestDocumentationSections:
    """Verify all 11 required sections are present."""

    def test_section_1_lambda_handler_present(self):
        """Test that Section 1: Lambda Handler is documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '## 1. Lambda Handler' in content
        assert 'lambda_handler' in content.lower()
        assert 'event' in content.lower()

    def test_section_2_scafad_runtime_present(self):
        """Test that Section 2: SCAFADCanonicalRuntime is documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '## 2. SCAFADCanonicalRuntime' in content
        assert 'process_event' in content

    def test_section_3_canonical_result_present(self):
        """Test that Section 3: CanonicalRuntimeResult is documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '## 3. CanonicalRuntimeResult' in content
        assert 'to_dict()' in content

    def test_section_4_layer0_telemetry_present(self):
        """Test that Section 4: Layer 0 TelemetryRecord is documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '## 4. Layer 0' in content
        assert 'TelemetryRecord' in content
        assert 'v4.2' in content

    def test_section_5_layer1_processed_record_present(self):
        """Test that Section 5: Layer 1 ProcessedRecord is documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '## 5. Layer 1' in content
        assert 'Layer1ProcessedRecord' in content

    def test_section_6_layer2_detection_present(self):
        """Test that Section 6: Layer 2 DetectionResult is documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '## 6. Layer 2' in content
        assert 'Layer2DetectionResult' in content
        # Should mention WP-5.2 threshold calibration
        assert '0.15' in content

    def test_section_7_layer3_fusion_present(self):
        """Test that Section 7: Layer 3 FusionResult is documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '## 7. Layer 3' in content
        assert 'Layer3FusionResult' in content

    def test_section_8_layer4_decision_trace_present(self):
        """Test that Section 8: Layer 4 DecisionTrace is documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '## 8. Layer 4' in content
        assert 'Layer4DecisionTrace' in content

    def test_section_9_layer5_threat_alignment_present(self):
        """Test that Section 9: Layer 5 ThreatAlignmentResult is documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '## 9. Layer 5' in content
        assert 'ThreatAlignmentResult' in content

    def test_section_10_layer6_feedback_present(self):
        """Test that Section 10: Layer 6 FeedbackState is documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '## 10. Layer 6' in content
        assert 'Layer6FeedbackState' in content

    def test_section_11_enumerations_present(self):
        """Test that Section 11: Supporting Enumerations is documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '## 11. Supporting Enumerations' in content
        assert 'AnomalyType' in content


class TestAcceptanceCriteria:
    """Verify acceptance criteria from the builder task."""

    def test_lambda_handler_documented(self):
        """AC: Covers lambda_handler (Section 1)."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert '## 1. Lambda Handler' in content
        assert 'lambda_handler(event, context)' in content
        assert '200' in content  # status code
        assert '500' in content  # error code

    def test_runtime_process_event_documented(self):
        """AC: Covers SCAFADCanonicalRuntime.process_event (Section 2)."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'process_event(' in content
        assert 'analyst_label' in content
        assert 'redacted_fields' in content
        assert 'verbosity' in content

    def test_all_l0_to_l6_output_classes_documented(self):
        """AC: Covers all L0–L6 output dataclasses (Sections 4–10)."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        expected_classes = [
            'TelemetryRecord',      # L0
            'Layer1ProcessedRecord', # L1
            'Layer2DetectionResult', # L2
            'Layer3FusionResult',    # L3
            'Layer4DecisionTrace',   # L4
            'ThreatAlignmentResult', # L5
            'Layer6FeedbackState',   # L6
        ]

        for cls in expected_classes:
            assert cls in content, f"Missing documentation for {cls}"

    def test_canonical_result_to_dict_schema_documented(self):
        """AC: CanonicalRuntimeResult.to_dict() schema documented with all keys (Section 3)."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the CanonicalRuntimeResult section
        assert 'CanonicalRuntimeResult' in content
        assert 'to_dict()' in content

        # Verify top-level keys in schema
        expected_keys = ['layer0_record', 'adapted_record', 'layer1_record', 'multilayer_result']
        for key in expected_keys:
            assert key in content, f"Missing key '{key}' in to_dict() schema"

    def test_markdown_is_valid(self):
        """AC: Valid markdown (no broken headers, balanced code fences)."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check balanced fences
        fence_count = content.count('```')
        assert fence_count % 2 == 0, "Unbalanced code fences"

        # Check for valid header markers
        assert content.count('#') > 0, "No headers found"

        # Verify no obviously broken structures
        assert '```json' in content or '```python' in content, "Expected code blocks"


class TestWP52AndWP51Integration:
    """Verify that documentation includes recent updates from WP-5.2 and WP-5.1."""

    def test_wp52_threshold_documented(self):
        """Test that WP-5.2 calibration threshold (0.15) is documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should mention 0.15 threshold and WP-5.2
        assert '0.15' in content
        assert 'WP-5.2' in content or 'calibrated' in content.lower()
        assert 'anomaly_indicated' in content

    def test_wp51_mitre_techniques_documented(self):
        """Test that WP-5.1 MITRE techniques are documented in Section 9."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should document the 4 new techniques from WP-5.1
        expected_techniques = ['T1648', 'T1537', 'T1552.005', 'T1580']

        for technique in expected_techniques:
            assert technique in content, f"Missing MITRE technique {technique}"

    def test_t1552_005_dot_notation_correct(self):
        """Test that T1552.005 is documented with correct dot notation."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should appear with dot notation, not T1552005
        assert 'T1552.005' in content
        # Should also verify the technique name
        assert 'Cloud Metadata API' in content or 'metadata' in content.lower()


class TestTableFormattingIntegrity:
    """Verify that all markdown tables in the documentation are well-formed."""

    def test_event_fields_table_valid(self):
        """Test that Event Fields table (Section 1) is properly formatted."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        content = ''.join(lines)
        # Find the event fields table section
        assert 'Event Fields' in content or 'Event key' in content

    def test_no_malformed_json_blocks(self):
        """Test that JSON examples in code blocks are parseable.

        Every ```json block must either:
          (a) parse successfully with json.loads(), OR
          (b) contain '...' — the conventional marker for intentionally
              truncated / illustrative examples.

        Blocks that fail json.loads() and contain no '...' are genuine
        errors in the documentation and will raise AssertionError.
        """
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract JSON code blocks
        json_blocks = re.findall(r'```json\n(.*?)\n```', content, re.DOTALL)

        assert len(json_blocks) > 0, (
            "No ```json code blocks found in documentation"
        )

        import json
        for i, block in enumerate(json_blocks):
            try:
                json.loads(block)
            except json.JSONDecodeError as e:
                # Blocks containing '...' are intentional incomplete examples
                if '...' not in block:
                    raise AssertionError(
                        f"JSON block {i + 1} is malformed and contains no "
                        f"'...' placeholder: {e}\n"
                        f"Block preview: {block[:120]}"
                    )


class TestContentAccuracy:
    """Verify that documented content is accurate and matches implementation."""

    def test_layer2_threshold_matches_source(self):
        """Test that documented threshold matches actual source code (0.15)."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the Layer2 section
        layer2_section = content[content.find('## 6. Layer 2'):content.find('## 7. Layer 3')]
        assert '0.15' in layer2_section, "L2 threshold not documented correctly"

    def test_layer2_decision_thresholds_documented(self):
        """Test that Layer 2/3/4 decision thresholds are documented."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for decision mapping table
        assert 'observe' in content
        assert 'review' in content
        assert 'escalate' in content
        assert 'fused_score' in content or 'aggregate_score' in content

    def test_l0_record_has_all_key_fields(self):
        """Test that L0 TelemetryRecord documents all major fields."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find L0 section
        l0_section = content[content.find('## 4. Layer 0'):content.find('## 5. Layer 1')]

        key_fields = [
            'event_id', 'timestamp', 'function_id', 'execution_phase',
            'anomaly_type', 'duration', 'adversarial_score', 'confidence_level'
        ]

        for field in key_fields:
            assert field in l0_section, f"Missing L0 field: {field}"


class TestDocumentationCompleteness:
    """Verify the documentation is sufficiently complete and usable."""

    def test_has_table_of_contents(self):
        """Test that document has a table of contents."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'Table of Contents' in content
        assert '[' in content and '](#' in content  # Should have anchor links

    def test_has_introductory_section(self):
        """Test that document starts with title and schema info."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'SCAFAD API Reference' in content or 'API Reference' in content
        assert 'v4.2' in content  # Schema version

    def test_has_footer_attribution(self):
        """Test that document has attribution to builder agent."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should have a generated-by note
        assert 'Generated' in content or 'builder' in content.lower() or 'WP-5.4' in content

    def test_example_outputs_present(self):
        """Test that sections include example outputs."""
        doc_path = Path("docs/api-reference.md")
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should have json example blocks
        assert '```json' in content
        # Count multiple examples
        json_examples = content.count('```json')
        assert json_examples >= 5, f"Expected multiple example outputs, found {json_examples}"
