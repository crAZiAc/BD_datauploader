# BDdatauploader

This project provides tools to upload and explore data in BlueDolphin using its public API.  
The main application is built with [Streamlit](https://streamlit.io).

## License
This project is licensed under the MIT License. See the MIT License.txt file for details.

## Contribution Guidelines
- All changes are made through pull requests.
- Direct commits to `main` require **review and approval by at least one other contributor**.
    - This is not enforced, but please notify other contributors to check your improvements.
- Use feature branches for development and testing.

## Running Locally

### Prerequisites
- Python 3.9+
- Streamlit

### Setup with virtual environment

1. Clone the repository and check out the branch you want to test:

   git fetch origin <branch-name>
   git checkout <branch-name>

2. Create and activate a virtual environment:

   Linux/macOS:
     python -m venv venv
     source venv/bin/activate

   Windows (PowerShell):
     python -m venv venv
     venv\Scripts\Activate.ps1

3. Install dependencies:

   pip install -r requirements.txt

4. Run the Streamlit app:

   streamlit run app.py

5. Open the URL shown in the terminal (usually http://localhost:8501) in your browser.

---

Happy uploading!
