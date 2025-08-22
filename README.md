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
- [Streamlit](https://docs.streamlit.io/library/get-started/installation)

### Setup with virtual environment
1. Clone the repository and check out the branch you want to test:
   
   git fetch origin <branch-name>
   git checkout <branch-name>

Create and activate a virtual environment:

  python -m venv venv
  # On Linux/macOS:
  source venv/bin/activate
  # On Windows (PowerShell):
  venv\Scripts\Activate.ps1

Install dependencies:

  pip install -r requirements.txt

Run the Streamlit app:

  streamlit run app.py
  Replace app.py with the entrypoint file of the Streamlit app if different.

Open the URL shown in the terminal (usually http://localhost:8501) in your browser.
