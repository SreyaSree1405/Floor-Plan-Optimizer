# Floor Plan Optimizer
Floor Plan Optimizer is a tool that takes a floor plan image as input and evaluates it based on architectural design principles. It scores each room and the overall layout on criteria such as space utilization, lighting and ventilation, accessibility, and functional layout. Additionally, it provides recommendations for improvements.

# Features
- Upload floor plan images.
- OCR (PyTesseract) to detect room names and boundaries.
- Compute area, room count, and number of windows.
- With the number of occupants and match against standards.
- Scoring system based on:
- Space utilization
     - Natural lighting and ventilation
     - Accessibility
     - Functional layout
- Suggestions based on architectural best practices.
- Visual output: annotated floor plans with scores and feedback.

# Running

1. Start the Flask backend:
   
      cd backend
      python app.py
     
2. Launch the frontend:

   - Open UploadUI.HTML in your browser.
     
3. Interact with the App:
   - Upload a floor plan image.
   - Select the number of occupants.
   - View room-wise and overall scores with recommendations.
   - Output visuals will be generated and displayed.

