from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Frame, PageTemplate, BaseDocTemplate, SimpleDocTemplate, PageBreak
import random

# ========== Layout ==========

def generate_doc(filename_or_buffer):
    """
    Generates a document object with a
    two-column layout.
    """
    # Column dimensions
    margin = 0.5 * inch
    gutter = 0.2 * inch
    col_width = (letter[0] - margin * 2 - gutter) / 2
    col_height = letter[1] - margin * 2

    left_column = Frame(
        x1=margin,
        y1=margin,
        width=col_width,
        height=col_height,
        id='left'
    )

    right_column = Frame(
        x1= margin + col_width + gutter,
        y1=margin,
        width=col_width,
        height=col_height,
        id='right'
    )

    two_column_template = PageTemplate(id='TwoColumns', frames=[left_column, right_column])

    doc = BaseDocTemplate(
        filename=filename_or_buffer, 
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin
    )

    # Give the document two columns
    doc.addPageTemplates([two_column_template])
    return doc

def generate_flashcard_doc(filename_or_buffer):
    """
    Generates a 5x3 inch flashcard
    ReportLab document.
    """
    # Flashcard dimensions
    margin = 0.2 * inch
    page_width = 5 * inch
    page_height = 3 * inch

    doc = SimpleDocTemplate(
        filename=filename_or_buffer,
        pagesize=(page_width, page_height),
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin
    )

    return doc

# ========== Styles ==========

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name='CheatsheetTitle',
    parent=styles['Heading1'],
    fontSize=20,
))

styles.add(ParagraphStyle(
    name='Concept',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=colors.darkblue,
    spaceAfter=0.1*inch,
    spaceBefore=0.3*inch
))

styles.add(ParagraphStyle(
    name='Subheader',
    fontSize=10,
    textColor=colors.darkred,
    spaceAfter=0.1*inch
))

styles.add(ParagraphStyle(
    name='Body',
    parent=styles['BodyText'],
    fontSize=10,
    spaceAfter=0.1*inch
))

styles.add(ParagraphStyle(
    name='FlashCardText',
    parent=styles['BodyText'],
    fontSize=14,
    leading=18,
    alignment=1
))

# ========== Story Building Functions ==========


def build_cheatsheet_story(title, concepts, definitions, equations, examples, references):
    """
    Generates a ReportLab-styles story
    returned as a list of Reportlab elements
    """
    story = []  # Create story content
    story.append(Paragraph(title, styles['CheatsheetTitle']))

    # Add content for each concept
    for concept in concepts:
        # Concept header
        story.append(Paragraph(concept, styles['Concept']))
        
        # Definition
        if concept in definitions:
            story.append(Paragraph("<b>Definition</b>", styles['Subheader']))
            story.append(Paragraph(definitions[concept], styles['Body']))
        
        # Equations
        if concept in equations and equations[concept]:
            story.append(Paragraph("<b>Equations</b>", styles['Subheader']))
            for eq in equations[concept]:
                story.append(Paragraph(eq, styles['Bullet']))
        
        # Examples
        if concept in examples and examples[concept]:
            story.append(Paragraph("<b>Examples</b>", styles['Subheader']))
            for ex in examples[concept]:
                story.append(Paragraph(ex, styles['Bullet']))

    # Add references like any good scientist
    story.append(Paragraph('<b>References</b>', styles['Concept']))
    for ref in references:
        story.append(Paragraph(ref, styles['Bullet']))
            
    return story

def build_quiz_story(title, questions, mapped_distractors, true_answers, references):
    """
    Generates a ReportLab story
    return as a list of ReportLab elements
    """
    # Store answers at the back 
    back_answers = []
    story = []
    
    story.append(Paragraph(title, styles['CheatsheetTitle']))

    if not (len(questions) == len(mapped_distractors) and len(mapped_distractors) == len(true_answers)):
        raise Exception('questions, distractors and true answers must all be of the same length')

    story.append(Paragraph('<b>Questions</b>', styles['Concept']))

    for question, distractors, true_answer in zip(questions, mapped_distractors, true_answers):
        shuffled = distractors + [true_answer]  # Not shuffled yet
        random.shuffle(shuffled)                # Now it is shuffled !
        shuffled = [f'{index + 1}. {answer}' for index, answer in enumerate(shuffled)]
        back_answers.append([a for a in shuffled if true_answer in a][0])

        story.append(Paragraph(question, styles['Body']))
        
        for ans in shuffled:
            story.append(Paragraph(ans, styles['Bullet']))

    # Send answers to the back (no peeking!)
    story.append(PageBreak())

    story.append(Paragraph('<b>Answers</b>', styles['Concept']))

    for question, ans in zip(questions, back_answers):
        story.append(Paragraph(question, styles['Body']))
        story.append(Paragraph(ans, styles['Bullet']))

    # Add references like any good scientist
    story.append(Paragraph('<b>References</b>', styles['Concept']))

    for ref in references:
        story.append(Paragraph(ref, styles['Bullet']))

    return story

def build_flashcards_story(questions, answers):
    """
    Generates PDF flashcards using
    ReportLab as question-answer pairs
    """
    story = []

    for q, a in zip(questions, answers):
        story.append(Paragraph(q, styles['FlashCardText']))
        story.append(PageBreak())
        story.append(Paragraph(a, styles['FlashCardText']))
        story.append(PageBreak())

    return story