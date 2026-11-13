"""
outputs/presentation/build_deck.py

Builds the complete 16-slide professional BTech CSE project presentation:
FruitFly-ML: Real-Time Detailed Video Analysis of Fruit Flies
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# 1. Color Palette Definitions (Dark Navy Modern Academic Theme)
COLOR_BG = RGBColor(10, 25, 47)         # Deep Navy (#0A192F)
COLOR_CARD = RGBColor(22, 34, 56)       # Slate Card Fill (#162238)
COLOR_CARD_ALT = RGBColor(30, 41, 59)   # Slate Card Secondary (#1E293B)
COLOR_BORDER = RGBColor(51, 65, 85)     # Slate Border (#334155)
COLOR_ACCENT = RGBColor(56, 189, 248)   # Sky Blue (#38BDF8)
COLOR_ACCENT2 = RGBColor(52, 211, 153)  # Emerald Green (#34D399)
COLOR_ACCENT3 = RGBColor(245, 158, 11)  # Amber (#F59E0B)
COLOR_WHITE = RGBColor(248, 250, 252)   # Pure White (#F8FAFC)
COLOR_MUTED = RGBColor(148, 163, 184)   # Muted Gray (#94A3B8)
COLOR_CYAN = RGBColor(6, 182, 212)      # Cyan (#06B6D4)

FONT_HEADING = "Segoe UI"
FONT_BODY = "Segoe UI"

def create_presentation():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_blank_slide_with_bg():
        slide = prs.slides.add_slide(blank_layout)
        # Background shape
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG
        bg.line.fill.background()
        return slide

    def add_header(slide, tag_text, title_text):
        # Category Tag
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.35))
        tf_tag = tag_box.text_frame
        tf_tag.word_wrap = True
        tf_tag.margin_left = tf_tag.margin_top = tf_tag.margin_right = tf_tag.margin_bottom = 0
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = tag_text.upper()
        p_tag.font.name = FONT_HEADING
        p_tag.font.size = Pt(11)
        p_tag.font.bold = True
        p_tag.font.color.rgb = COLOR_ACCENT

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.733), Inches(0.6))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = FONT_HEADING
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_WHITE

        # Divider line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.42), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = COLOR_BORDER
        line.line.fill.background()

    def add_card(slide, left, top, width, height, fill_color=COLOR_CARD, border_color=COLOR_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = fill_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
        return card

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    slide1 = add_blank_slide_with_bg()

    # Category tag
    tag_box1 = slide1.shapes.add_textbox(Inches(1.0), Inches(0.9), Inches(11.333), Inches(0.4))
    p = tag_box1.text_frame.paragraphs[0]
    p.text = "B.TECH COMPUTER SCIENCE & ENGINEERING — CAPSTONE PROJECT DEFENSE"
    p.font.name = FONT_HEADING
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    # Main Title
    title_box1 = slide1.shapes.add_textbox(Inches(1.0), Inches(1.35), Inches(11.333), Inches(1.4))
    tf1 = title_box1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Real-Time Detailed Video Analysis of Fruit Flies"
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE

    p2 = tf1.add_paragraph()
    p2.text = "Automated Multi-Attribute Behavioral Phenotyping using Classical Machine Learning"
    p2.font.name = FONT_HEADING
    p2.font.size = Pt(18)
    p2.font.color.rgb = COLOR_ACCENT2

    # Left Info Card: Project Foundations
    c1 = add_card(slide1, Inches(1.0), Inches(3.2), Inches(5.4), Inches(3.6))
    tf_c1 = c1.text_frame
    tf_c1.word_wrap = True
    tf_c1.margin_left = tf_c1.margin_top = tf_c1.margin_right = tf_c1.margin_bottom = Inches(0.3)
    
    p = tf_c1.paragraphs[0]
    p.text = "PROJECT METADATA & FOUNDATION"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    items_c1 = [
        ("Baseline Reference:", "Stanford CS229 Project (Herbst & Candelier)"),
        ("Domain:", "Computer Vision & Quantitative Ethology"),
        ("Core Capabilities:", "Detection, Sex ID, 360° Heading Pose, Courtship Wing Angles"),
        ("Architecture:", "Deterministic CPU Pipeline (Zero GPU Dependency)"),
        ("Verified Throughput:", "12.11 FPS (1530x1530 Resolution on CPU)")
    ]
    for lbl, val in items_c1:
        p = tf_c1.add_paragraph()
        p.text = f"• {lbl} {val}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(8)

    # Right Info Card: Student & Institutional Placeholders
    c2 = add_card(slide1, Inches(6.9), Inches(3.2), Inches(5.4), Inches(3.6))
    tf_c2 = c2.text_frame
    tf_c2.word_wrap = True
    tf_c2.margin_left = tf_c2.margin_top = tf_c2.margin_right = tf_c2.margin_bottom = Inches(0.3)

    p = tf_c2.paragraphs[0]
    p.text = "CANDIDATE & INSTITUTIONAL DETAILS"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    items_c2 = [
        ("Student Presenter(s):", "[Student Name(s) | Roll Number(s)]"),
        ("Project Supervisor / Guide:", "[Project Guide Name, Designation]"),
        ("Department:", "[Department of Computer Science & Engineering]"),
        ("Institution:", "[College / University Name]"),
        ("Academic Year:", "[2025 – 2026]"),
        ("Evaluation Panel:", "[External Examiner & Department Committee]")
    ]
    for lbl, val in items_c2:
        p = tf_c2.add_paragraph()
        p.text = f"• {lbl} {val}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 2: Abstract & Motivation
    # =========================================================================
    slide2 = add_blank_slide_with_bg()
    add_header(slide2, "1. Research Background & Motivation", "Automated Quantitative Ethology in Drosophila Melanogaster")

    card2_1 = add_card(slide2, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2))
    tf = card2_1.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.35)
    
    p = tf.paragraphs[0]
    p.text = "The Biological Challenge"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT3

    points_2_1 = [
        "Fundamental Model Organism: Drosophila melanogaster is central to behavioral genetics, neuroscience, and pharmacology.",
        "Courtship & Social Dynamics: High-speed interactions (wing vibrations, copulation rituals, chasing) occur in sub-second bursts.",
        "Manual Video Bottlenecks: Human review of 35 FPS video requires 100+ hours per trial, suffering from observer fatigue and subjective bias.",
        "Flaws of Invasive Markers: Physical tagging (fluorescent dots, paint, tethering) alters natural locomotion, flight aerodynamic drag, and courtship acceptance."
    ]
    for pt in points_2_1:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(12)

    card2_2 = add_card(slide2, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tf = card2_2.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.35)

    p = tf.paragraphs[0]
    p.text = "The FruitFly-ML Solution"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT2

    points_2_2 = [
        "Non-Invasive High-Throughput Vision: Automated markerless analysis directly on standard high-resolution overhead video.",
        "Four Simultaneous Phenotypes: Extracts (1) Chamber noise filtering, (2) Sex identification, (3) 360° heading direction, (4) Courtship wing display angles.",
        "Sub-Millisecond CPU Inference: Pure ML inference takes < 0.10 ms per fly (> 10,000 samples/sec), eliminating the requirement for expensive GPUs.",
        "Data-Centric Rigor: Evaluated using GroupShuffleSplit to guarantee zero data leakage between augmented pairs and source video frames."
    ]
    for pt in points_2_2:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(12)

    # =========================================================================
    # SLIDE 3: Problem Statement & Objectives
    # =========================================================================
    slide3 = add_blank_slide_with_bg()
    add_header(slide3, "2. Problem Definition & Scope", "Technical Challenges & Core Engineering Objectives")

    col_w = Inches(3.64)
    c3_1 = add_card(slide3, Inches(0.8), Inches(1.7), col_w, Inches(5.2))
    c3_2 = add_card(slide3, Inches(4.84), Inches(1.7), col_w, Inches(5.2))
    c3_3 = add_card(slide3, Inches(8.88), Inches(1.7), col_w, Inches(5.2))

    for card, title, color, pts in [
        (c3_1, "1. Spatial Challenges", COLOR_ACCENT, [
            "Chamber Clutter: Distinguish flies from chamber dust, perimeter reflections, and shadows.",
            "Axial Symmetry Ambiguity: Standard second-order moments calculate body axis in [-90°, +90°], unable to distinguish head from tail (180° ambiguity).",
            "Variable Body Posture: Flies twist and groom, altering apparent contour shape."
        ]),
        (c3_2, "2. Occlusion & Morphology", COLOR_ACCENT3, [
            "Physical Interaction & Mating: When flies touch, adaptive thresholding forms a merged single contour.",
            "Sexual Dimorphism: Males and females exhibit subtle area and aspect ratio differences that must be separated robustly.",
            "Semi-Translucent Wings: Wing membranes have faint pixel contrast, requiring sensitive morphological thresholding."
        ]),
        (c3_3, "3. Engineering Objectives", COLOR_ACCENT2, [
            "100% Deterministic CPU Flow: Guarantee execution on commodity CPUs without cloud or GPU dependencies.",
            "Zero Data Leakage: Prove mathematical independence of test evaluation through strict frame grouping.",
            "Full Telemetry Output: Generate synchronized 10-column trajectory logs and annotated video outputs."
        ])
    ]:
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.3)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = color
        for pt in pts:
            p = tf.add_paragraph()
            p.text = "• " + pt
            p.font.size = Pt(11.5)
            p.font.color.rgb = COLOR_WHITE
            p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 4: Stanford CS229 Baseline vs Our Contributions
    # =========================================================================
    slide4 = add_blank_slide_with_bg()
    add_header(slide4, "3. Comparative Literature Baseline", "Stanford CS229 Baseline vs. FruitFly-ML Contributions")

    c4_1 = add_card(slide4, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2))
    tf1 = c4_1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = Inches(0.35)
    
    p = tf1.paragraphs[0]
    p.text = "Stanford CS229 Baseline (2014)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_MUTED

    stan_pts = [
        "Legacy Environment: Implemented for Python 2.7, OpenCV 2/3 on 32-bit Linux; incompatible with modern runtime stacks.",
        "Naive Random Splitting: Evaluated using standard train_test_split. Augmented pairs (0° and 180° rot) from the same frame leaked across train/test splits.",
        "Video Decoder Truncation: Relied on Linux FFmpeg demuxing; halts early on tail GOP timestamps, losing 215 frames on Windows.",
        "Testing & Robustness: Contained no automated test assertions, regression suites, or exception handling for zero-moment divide-by-zero."
    ]
    for pt in stan_pts:
        p = tf1.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(10)

    c4_2 = add_card(slide4, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tf2 = c4_2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = Inches(0.35)

    p = tf2.paragraphs[0]
    p.text = "FruitFly-ML Contributions (Our Work)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    our_pts = [
        "Modernized Architecture: Full migration to Python 3.13, OpenCV 5 API, and headless automated environments on Windows 64-bit.",
        "Zero Data Leakage (GroupShuffleSplit): Grouping samples by source frame guarantees correlated augmented pairs never leak across splits.",
        "Complete Video Stream Recovery: Windows Media Foundation (CAP_MSMF) + deduplication achieves 100% decoding (2,918 / 2,918 frames).",
        "Comprehensive Automated Testing: 35-test pytest suite, 5 verification scripts, and 7 defensive bug fixes protecting against runtime crashes."
    ]
    for pt in our_pts:
        p = tf2.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 5: Proposed FruitFly-ML System
    # =========================================================================
    slide5 = add_blank_slide_with_bg()
    add_header(slide5, "4. Proposed System Framework", "The FruitFly-ML Processing Framework")

    # Embed architecture diagram image
    arch_img = "outputs/presentation/presentation_assets/architecture_diagram.png"
    if os.path.exists(arch_img):
        slide5.shapes.add_picture(arch_img, Inches(0.8), Inches(1.65), Inches(11.733), Inches(3.6))

    # 3 Summary Cards below image
    c5_1 = add_card(slide5, Inches(0.8), Inches(5.45), Inches(3.64), Inches(1.55))
    tf = c5_1.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = "Deterministic Ingestion"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    p = tf.add_paragraph()
    p.text = "35.0 FPS hardware capture, circular chamber masking (R=680 px), and dual-stage morphological thresholding."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_WHITE

    c5_2 = add_card(slide5, Inches(4.84), Inches(5.45), Inches(3.64), Inches(1.55))
    tf = c5_2.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = "Sub-Millisecond Inference"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT2
    p = tf.add_paragraph()
    p.text = "4 Classical ML models operate in < 0.10 ms per fly, extracting sex, pose, and wing angles without deep learning overhead."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_WHITE

    c5_3 = add_card(slide5, Inches(8.88), Inches(5.45), Inches(3.64), Inches(1.55))
    tf = c5_3.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = "Synchronized Outputs"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT3
    p = tf.add_paragraph()
    p.text = "Full-resolution 1530x1530 annotated MP4 video + synchronized 10-column physical trajectory CSV logs with null safety."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_WHITE

    # =========================================================================
    # SLIDE 6: Overall System Architecture & State Machine
    # =========================================================================
    slide6 = add_blank_slide_with_bg()
    add_header(slide6, "5. Architecture & State Management", "State Machine & Multi-Stage Video Execution Flow")

    c6_1 = add_card(slide6, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2))
    tf = c6_1.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.35)

    p = tf.paragraphs[0]
    p.text = "The 4 Operational Tracking States"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    states = [
        ("TWO_FLIES_SEPARATED (Normal State)", "Exactly 2 individual contours detected. Executes complete inference: Sex ID -> 360° Pose -> Courtship Wing Regression. Full cyan/magenta overlay rendered."),
        ("OVERLAPPING_FLIES (Occlusion State)", "One or more contours exceed 10,131 px. Renders amber boundary and suppresses individual angles to prevent synthetic data fabrication during mating."),
        ("SINGLE_FLY_DETECTED (Fallback State)", "Only 1 contour detected. Partial tracking logged; second fly marked as temporarily occluded."),
        ("NO_FLIES_DETECTED (Chamber Glare)", "0 contours detected. Safe null row logged; zero crashes.")
    ]
    for st, desc in states:
        p = tf.add_paragraph()
        p.text = "• " + st + ":"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT2 if "SEPARATED" in st else (COLOR_ACCENT3 if "OVERLAPPING" in st else COLOR_MUTED)
        p.space_before = Pt(8)
        p_sub = tf.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10)
        p_sub.font.color.rgb = COLOR_WHITE

    c6_2 = add_card(slide6, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tf = c6_2.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.35)

    p = tf.paragraphs[0]
    p.text = "Pipeline Execution Lifecycle"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT3

    stages = [
        ("Stage 1: Stream Ingestion", "VideoCapture opens MP4 via Media Foundation; duplicate frames filtered via np.array_equal."),
        ("Stage 2: Chamber Segmentation", "Static binary circle mask applied; fixed threshold (115) extracts dark chitin cores."),
        ("Stage 3: Decision Tree Dispatch", "Contour area classifies clutter vs single fly vs merged pair; determines state."),
        ("Stage 4: Modular ML Inference", "Decoupled models compute sex, moments, HOG, PCA, and regression sequentially."),
        ("Stage 5: Output Generation", "OpenCV VideoWriter encodes annotated MP4; CSV writer flushes frame-by-frame.")
    ]
    for stg, desc in stages:
        p = tf.add_paragraph()
        p.text = "• " + stg + ":"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        p.space_before = Pt(8)
        p_sub = tf.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10)
        p_sub.font.color.rgb = COLOR_WHITE

    # =========================================================================
    # SLIDE 7: Dataset & LabelMe Annotation Engineering
    # =========================================================================
    slide7 = add_blank_slide_with_bg()
    add_header(slide7, "6. Dataset Engineering", "Ground Truth Dataset & Anatomical Landmark Protocol")

    c7_1 = add_card(slide7, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2))
    tf = c7_1.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.35)

    p = tf.paragraphs[0]
    p.text = "Chamber Dataset Composition"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    ds_info = [
        "326 High-Resolution Images: Exactly 1530 x 1530 px captured under standardized circular chamber lighting.",
        "5 Independent Recording Sessions: 12-04_17-54-43 (100) | 12-05_12-43-00 (33) | 12-07_16-45-00 (50) | 12-08_11-15-00 (102) | 12-08_22-00-00 (41).",
        "LabelMe JSON Annotations: Manual ground truth point coordinates created by Stanford neurobiology researchers.",
        "100% Filesystem Integrity: Re-verified via test_annotations.py (326 / 326 images verified, 0 missing, 0 corrupt)."
    ]
    for pt in ds_info:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(10)

    c7_2 = add_card(slide7, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tf = c7_2.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.35)

    p = tf.paragraphs[0]
    p.text = "Anatomical Landmark Protocol"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT2

    landmarks = [
        ("mp / fp (Positions)", "Thorax center-of-mass for male and female flies."),
        ("mh / fh (Heads)", "Center of head between eyes, defining anterior vector."),
        ("ma / fa (Abdomens)", "Posterior abdomen tip, defining posterior vector."),
        ("mlw / mrw (Wingtips)", "Male left and right wingtips for courtship angle estimation."),
        ("Quality Filter (check_features)", "Only complete landmark sets enter model training (239 male pose, 174 female pose, 236 male wing frames). Incomplete frames filtered out safely.")
    ]
    for lm, desc in landmarks:
        p = tf.add_paragraph()
        p.text = "• " + lm + ":"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        p.space_before = Pt(6)
        p_sub = tf.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10)
        p_sub.font.color.rgb = COLOR_WHITE

    # =========================================================================
    # SLIDE 8: Preprocessing, Feature Extraction & Augmentation
    # =========================================================================
    slide8 = add_blank_slide_with_bg()
    add_header(slide8, "7. Feature Engineering", "Image Moments, HOG Gradients & Physical Invariances")

    c8_1 = add_card(slide8, Inches(0.8), Inches(1.7), col_w, Inches(5.2))
    c8_2 = add_card(slide8, Inches(4.84), Inches(1.7), col_w, Inches(5.2))
    c8_3 = add_card(slide8, Inches(8.88), Inches(1.7), col_w, Inches(5.2))

    for card, title, color, pts in [
        (c8_1, "1. Moments & Inertia", COLOR_ACCENT, [
            "Green's Theorem Area: Area = 0.5 * oint(x dy - y dx).",
            "Normalized Central Moments: mu_pq computes spatial inertia ellipse axes.",
            "Aspect Ratio (MA/ma): Exploits sexual dimorphism; females are larger and more elongated.",
            "Initial Axis Alignment: Computes orientation in [-pi/2, +pi/2]."
        ]),
        (c8_2, "2. HOG Feature Space", COLOR_ACCENT3, [
            "Normalized Patches: Scaled to 96x80 pixels.",
            "Gradient Descriptors: Cell size 8x8, block size 16x16, 9 unsigned bins.",
            "Heading Orientation: 3,780 dense gradient features capture eye and abdomen texture.",
            "Wing Display: 3,564 features capture translucent wing edges."
        ]),
        (c8_3, "3. Physical Invariances", COLOR_ACCENT2, [
            "180° Rotational Augmentation: Duplicates patches and rotates 180°, doubling pose data (Male: 239 -> 478, Female: 174 -> 348).",
            "Bilateral Wing Reflection: Flips right wings horizontally across body midline (236 -> 472 samples).",
            "Coordinate Invariance: Projects both wings into a unified canonical space."
        ])
    ]:
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.3)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = color
        for pt in pts:
            p = tf.add_paragraph()
            p.text = "• " + pt
            p.font.size = Pt(11)
            p.font.color.rgb = COLOR_WHITE
            p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 9: Component 1 & 2: Fly Counting & Sex Identification
    # =========================================================================
    slide9 = add_blank_slide_with_bg()
    add_header(slide9, "8. Model Deep Dive (Part 1)", "Detection Decision Tree & Dimorphic Sex Classifier")

    c9_1 = add_card(slide9, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2))
    tf = c9_1.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.35)

    p = tf.paragraphs[0]
    p.text = "Model 1: FlyCount (Decision Tree)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    dt_info = [
        ("Input Feature:", "Scalar contour area A (Green's Theorem)."),
        ("Model Architecture:", "DecisionTreeClassifier (criterion='entropy', max_depth=2)."),
        ("Learned Split 1:", "A <= 1,413.5 px -> Class 'neither' (Noise, chamber debris, shadows)."),
        ("Learned Split 2:", "1,413.5 < A <= 10,131.0 px -> Class 'one' (Individual isolated fly)."),
        ("Learned Split 3:", "A > 10,131.0 px -> Class 'both' (Touching / overlapping mating pair)."),
        ("Test Performance:", "100.0% Test Accuracy on 252 test samples (0% Error)."),
        ("Inference Latency:", "0.0012 ms/sample (> 800,000 checks/sec).")
    ]
    for lbl, val in dt_info:
        p = tf.add_paragraph()
        p.text = f"• {lbl} {val}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(6)

    c9_2 = add_card(slide9, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tf = c9_2.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.35)

    p = tf.paragraphs[0]
    p.text = "Model 2: Sex Identification (Logistic Regression)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT2

    id_info = [
        ("Input Features:", "4D vector X = [Area_1, Area_2, Aspect_1, Aspect_2]."),
        ("Model Architecture:", "Pipeline(StandardScaler(), LogisticRegression(C=1.0))."),
        ("Biological Basis:", "Females have significantly larger abdominal mass and distinct elongation for egg carriage; males are smaller with rounded abdomens."),
        ("Classification Target:", "Binary: 0 -> [Male=c1, Female=c2], 1 -> [Female=c1, Male=c2]."),
        ("Test Performance:", "100.0% Test Accuracy on 142 test samples (0% Error)."),
        ("Inference Latency:", "0.0050 ms/sample (> 200,000 classifications/sec).")
    ]
    for lbl, val in id_info:
        p = tf.add_paragraph()
        p.text = f"• {lbl} {val}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 10: Component 3 & 4: Heading Orientation & Wing Regression
    # =========================================================================
    slide10 = add_blank_slide_with_bg()
    add_header(slide10, "9. Model Deep Dive (Part 2)", "360° Heading Pose Estimation & Courtship Wing Regression")

    c10_1 = add_card(slide10, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2))
    tf = c10_1.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.35)

    p = tf.paragraphs[0]
    p.text = "Model 3: 360° Heading Orientation"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    orient_info = [
        ("The 180° Ambiguity:", "Moments define axis in [-90°, +90°]. Machine learning resolves head vs abdomen."),
        ("Feature Pipeline:", "Fly patch -> Vertical alignment -> HOG (3,780D) -> PCA(15) -> Logistic Regression."),
        ("PCA Compression:", "15 principal components retain > 60% variance, eliminating overfitting."),
        ("Dedicated Sex Models:", "clf_orient_male & clf_orient_female specialize in sex-specific abdominal textures (male dark genital arch vs female striped tip)."),
        ("Test Performance:", "100.0% Male (120 test) | 100.0% Female (88 test)."),
        ("Inference Latency:", "0.026 - 0.034 ms/sample.")
    ]
    for lbl, val in orient_info:
        p = tf.add_paragraph()
        p.text = f"• {lbl} {val}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(6)

    c10_2 = add_card(slide10, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tf = c10_2.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.35)

    p = tf.paragraphs[0]
    p.text = "Model 4: Male Wing-Angle Regression"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT2

    wing_info = [
        ("Biological Phenomenon:", "Males extend a wing horizontally at ~90° during courtship to sing to females."),
        ("Bilateral Symmetry:", "Right wing horizontally reflected across body midline, projecting both wings into canonical left-wing space."),
        ("Feature Pipeline:", "Thorax patch (400x400) -> HOG (3,564D) -> PCA(40) -> Linear Regression (OLS)."),
        ("PCA Compression:", "40 components retain > 75% variance."),
        ("Test MAE & RMSE:", "MAE = 1.81° (0.0316 rad) | RMSE = 2.34° (0.0409 rad)."),
        ("Residual Std (sigma):", "sigma = 2.30° (0.0402 rad) | R² = 0.990 (99.0% variance)."),
        ("Inference Latency:", "0.0326 ms/sample.")
    ]
    for lbl, val in wing_info:
        p = tf.add_paragraph()
        p.text = f"• {lbl} {val}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 11: Training Methodology & GroupShuffleSplit Anti-Leakage
    # =========================================================================
    slide11 = add_blank_slide_with_bg()
    add_header(slide11, "10. Scientific Rigor & Validation", "Preventing Data Leakage via GroupShuffleSplit")

    # Embed GroupShuffleSplit diagram image
    gss_img = "outputs/presentation/presentation_assets/group_shuffle_split_diagram.png"
    if os.path.exists(gss_img):
        slide11.shapes.add_picture(gss_img, Inches(0.8), Inches(1.65), Inches(11.733), Inches(3.6))

    # 2 Cards below image explaining details
    c11_1 = add_card(slide11, Inches(0.8), Inches(5.45), Inches(5.6), Inches(1.55))
    tf = c11_1.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = "Why Naive Random Splitting Fails"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = RGBColor(239, 68, 68)
    p = tf.add_paragraph()
    p.text = "Random splitting distributes augmented pairs from the same physical frame into both train and test sets. The model simply memorizes the background camera noise and fly posture, inflating test accuracy."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_WHITE

    c11_2 = add_card(slide11, Inches(6.8), Inches(5.45), Inches(5.7), Inches(1.55))
    tf = c11_2.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = "GroupShuffleSplit Guarantee (Our Implementation)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT2
    p = tf.add_paragraph()
    p.text = "Grouping by group_id = session + frame ensures all derived variants exist strictly in train OR test. Scalers and PCA matrices are fitted strictly on train groups. Guarantees 100% genuine generalization."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_WHITE

    # =========================================================================
    # SLIDE 12: Experimental Results & Benchmark Verification
    # =========================================================================
    slide12 = add_blank_slide_with_bg()
    add_header(slide12, "11. Experimental Results", "Evaluation Metrics & Benchmark Comparison")

    # Table on Left
    table_shape = slide12.shapes.add_table(6, 4, Inches(0.8), Inches(1.7), Inches(6.2), Inches(5.2))
    table = table_shape.table
    table.columns[0].width = Inches(1.8)
    table.columns[1].width = Inches(1.5)
    table.columns[2].width = Inches(1.5)
    table.columns[3].width = Inches(1.4)

    table_data = [
        ["Model Task", "Stanford Baseline", "FruitFly-ML (Grouped)", "FruitFly-ML (Naive)"],
        ["FlyCount", "Error: 0.0%", "Error: 0.0% (100%)", "Error: 0.0% (100%)"],
        ["Sex ID", "Error: 0.0%", "Error: 0.0% (100%)", "Error: 0.0% (100%)"],
        ["Male Orient", "Error: 0.0%", "Error: 0.0% (100%)", "Error: 0.0% (100%)"],
        ["Female Orient", "Error: ~1.1%", "Error: 0.0% (100%)", "Error: 1.15%"],
        ["Wing Angle", "sigma ~ 2.35°", "sigma = 2.30° | MAE: 1.81°", "sigma = 3.00° | MAE: 2.10°"]
    ]

    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.font.name = FONT_HEADING
            p.font.size = Pt(9.5 if r_idx > 0 else 10.5)
            p.font.bold = (r_idx == 0 or c_idx == 2)
            p.font.color.rgb = COLOR_WHITE
            if r_idx == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(2, 136, 209)
            elif c_idx == 2:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(6, 95, 70)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_CARD if r_idx % 2 == 1 else COLOR_CARD_ALT

    # Embed Results Comparison Chart on Right
    chart_img = "outputs/presentation/presentation_assets/results_comparison_chart.png"
    if os.path.exists(chart_img):
        slide12.shapes.add_picture(chart_img, Inches(7.3), Inches(1.7), Inches(5.233), Inches(5.2))

    # =========================================================================
    # SLIDE 13: Video Inference Pipeline & Demonstration
    # =========================================================================
    slide13 = add_blank_slide_with_bg()
    add_header(slide13, "12. Real-Time Video Demonstration", "Video Inference Engine & Real-World Behavioral Screenshots")

    # 3 Screenshots side by side
    sc_w = Inches(3.64)
    sc_h = Inches(3.64)
    sc_top = Inches(1.7)

    sc_files = [
        ("outputs/screenshots/screenshot_two_flies_separated.jpg", "1. TWO_FLIES_SEPARATED", "Cyan Male, Magenta Female, direction arrows, head/abdomen dots, instantaneous angles.", Inches(0.8)),
        ("outputs/screenshots/screenshot_wing_angles_active.jpg", "2. ACTIVE COURTSHIP WINGS", "Yellow wing vectors projecting from thorax, real-time angular displays (L: 24°, R: 8°).", Inches(4.84)),
        ("outputs/screenshots/screenshot_overlapping_flies.jpg", "3. OVERLAPPING / MATING", "Amber boundary, CONTACT / OVERLAP label, null-safe telemetry preventing angle fabrication.", Inches(8.88))
    ]

    for fp, title, caption, left in sc_files:
        if os.path.exists(fp):
            slide13.shapes.add_picture(fp, left, sc_top, sc_w, sc_h)
        card = add_card(slide13, left, Inches(5.45), sc_w, Inches(1.55))
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.18)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        p = tf.add_paragraph()
        p.text = caption
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_WHITE

    # =========================================================================
    # SLIDE 14: Engineering Challenges & Robust Solutions
    # =========================================================================
    slide14 = add_blank_slide_with_bg()
    add_header(slide14, "13. Engineering Modernization", "Core Software Engineering Challenges & Solutions")

    grid_w = Inches(5.6)
    grid_h = Inches(2.45)

    challenges = [
        ("1. OpenCV 5 API Incompatibility", "cv2.findContours returned 3 values in OpenCV 3 but returns 2 in OpenCV 4/5.", "Implemented a version-agnostic contour unpacker (res[1] if len(res)==3 else res[0]) preventing unpack crashes.", Inches(0.8), Inches(1.7), COLOR_ACCENT),
        ("2. Video Truncation & GOP Padding", "Default Linux FFmpeg halted on tail GOP timestamps, dropping 215 frames.", "Migrated to Windows Media Foundation (CAP_MSMF) + np.array_equal duplicate filter, recovering 100% of frames (2,918 / 2,918).", Inches(6.8), Inches(1.7), COLOR_ACCENT3),
        ("3. Zero-Moment Divide-by-Zero", "Blank/edge patches caused m00 = 0, crashing spatial moment normalization.", "Engineered defensive guards in patch.py returning (0.0, 0.0) and aspect ratio 1.0, eliminating math crashes.", Inches(0.8), Inches(4.45), COLOR_ACCENT2),
        ("4. Headless Automated Testing", "OpenCV GUI calls (imshow) crash headless CI and test environments.", "Enforced Matplotlib Agg backend, isolated MPLCONFIGDIR, and designed a 35-test pytest suite.", Inches(6.8), Inches(4.45), COLOR_CYAN)
    ]

    for title, prob, sol, left, top, color in challenges:
        c = add_card(slide14, left, top, grid_w, grid_h)
        tf = c.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.25)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = color
        p = tf.add_paragraph()
        p.text = "• Problem: " + prob
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_MUTED
        p.space_before = Pt(4)
        p = tf.add_paragraph()
        p.text = "• Solution: " + sol
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(4)

    # =========================================================================
    # SLIDE 15: Limitations, Future Scope & Conclusion
    # =========================================================================
    slide15 = add_blank_slide_with_bg()
    add_header(slide15, "14. Summary & Future Scope", "Project Achievements, Limitations & Future Roadmap")

    c15_1 = add_card(slide15, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2))
    tf1 = c15_1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = Inches(0.35)

    p = tf1.paragraphs[0]
    p.text = "Project Summary & Achievements"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT2

    achieve = [
        "Faithful Modernization: Successfully modernized Stanford CS229 baseline for Python 3.13 and OpenCV 5.",
        "Deterministic CPU Performance: Proved that classical ML delivers microsecond inference (< 0.1 ms) and high accuracy without GPU compute.",
        "Zero Data Leakage: Verified 100% classification test accuracy and 1.81° wing MAE under rigorous GroupShuffleSplit evaluation.",
        "Full Telemetry & Video: Processed 2,918 frames across 5 videos with 100% 1:1 synchronization."
    ]
    for pt in achieve:
        p = tf1.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(10)

    c15_2 = add_card(slide15, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tf2 = c15_2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = Inches(0.35)

    p = tf2.paragraphs[0]
    p.text = "Limitations & Future Roadmap"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    future = [
        "Current Limitation: Physical mating merges contours, temporarily suspending individual pose until separation.",
        "Kalman Filter Tracking: Integrate constant-velocity Kalman filters with Hungarian matching for multi-second occluded ID persistence.",
        "Hybrid Deep Learning: Trigger lightweight pose regression micro-nets selectively ONLY during overlapping states.",
        "3D Volumetric Tracking: Calibrate multi-camera camera rigs for 3D flight saccade reconstruction."
    ]
    for pt in future:
        p = tf2.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 16: References & Acknowledgements
    # =========================================================================
    slide16 = add_blank_slide_with_bg()
    add_header(slide16, "15. Academic Citations & Acknowledgements", "References & Acknowledgements")

    c16_1 = add_card(slide16, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2))
    tf1 = c16_1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = Inches(0.35)

    p = tf1.paragraphs[0]
    p.text = "Primary Academic References"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    refs = [
        "Herbst, S., & Candelier, R. (2014). Real-time Detailed Video Analysis of Fruit Flies. Stanford CS229 Project.",
        "Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. IEEE CVPR.",
        "Jolliffe, I. T. (2002). Principal Component Analysis. Springer Series in Statistics.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research.",
        "Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools."
    ]
    for r in refs:
        p = tf1.add_paragraph()
        p.text = f"[{refs.index(r)+1}] {r}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(8)

    c16_2 = add_card(slide16, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tf2 = c16_2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = Inches(0.35)

    p = tf2.paragraphs[0]
    p.text = "Acknowledgements"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT2

    ack = [
        "Stanford CS229 Course Staff & Original Authors: For open-sourcing the biological dataset and classical computer vision pipeline architecture.",
        "Project Guide / Supervisor: [Project Guide Name, Designation] for ongoing mentorship, technical reviews, and engineering direction.",
        "Department of Computer Science & Engineering: [Institution Name] for computing infrastructure and academic resources.",
        "Project Evaluation Committee & Faculty: For their review, questions, and critical evaluation of this capstone defense."
    ]
    for a in ack:
        p = tf2.add_paragraph()
        p.text = "• " + a
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(10)

    # Save presentation
    out_pptx = "outputs/presentation/FruitFly_ML_Project_Presentation.pptx"
    prs.save(out_pptx)
    print("SUCCESS: Saved 16-slide presentation to:", out_pptx)

if __name__ == '__main__':
    create_presentation()

