# Machine Learning Prediction Server

## Overview

This project implements a Flask-based API server that serves predictions from a RandomForestClassifier model trained on user data to predict the number of purchases based on features: `age`, `gender`, `income`, `days_on_platform`, and `city`. The server preprocesses incoming data, applies the trained model, and returns predictions. It includes API key authentication and is deployed on an Azure VM using `gunicorn` as a WSGI server, configured to run as a systemd service for automatic startup.

## Project Structure

```
deployed-ml-server/
├── data/
│   └── data.csv                # Input dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessor.py    # Data preprocessing logic
│   ├── model_trainer.py        # Model training logic
├── models/
│   └── classification_model.pkl # Trained model
├── .env                        # Environment variables (API key)
├── .gitignore                  # Git ignore file
├── app.py                      # Flask API server
├── requirements.txt            # Python dependencies
└── train_model.py              # Script to train and save model
```

## Prerequisites

- **Azure Account**: Active subscription for creating a VM.
- **Local Machine**: Python 3.8+, Git, SSH client (e.g., OpenSSH or PuTTY), Postman for testing.
- **VM OS**: Ubuntu Server 20.04 LTS.
- **Data**: `data.csv` with columns `age`, `gender`, `income`, `days_on_platform`, `city`, `purchases`.

## Setup Instructions

### Step 1: Create Azure VM

1. **Log in to Azure Portal**: Go to portal.azure.com.
2. **Create Virtual Machine**:
   - Click **Create a resource** &gt; **Virtual machine**.
   - Configure:
     - **Subscription**: Your subscription.
     - **Resource group**: Create new (e.g., `ml-server-rg`).
     - **Name**: `ml-server-vm`.
     - **Region**: Choose a nearby region (e.g., East US).
     - **Image**: `Ubuntu Server 20.04 LTS - Gen2`.
     - **Size**: `Standard_B2s` ++(2 vCPUs, 4GB RAM).
     - **Authentication**: SSH public key.
       - **Username**: `azureuser`.
       - Generate & Download SSH key locally:
     - **Public inbound ports**: Allow **SSH (22)**.
   - Click **Review + create** &gt; **Create**.
3. **Note Public IP**: From the VM overview, note the public IP (e.g., `20.123.45.67`).
4. **Configure Networking**:
   - Go to VM’s **Networking** &gt; **Add inbound port rule**:
     - **Destination port ranges**: `5000`
     - **Protocol**: TCP
     - **Action**: Allow
     - **Name**: `AllowFlaskPort5000`
   - Save the rule.

### Step 2: Set Up VM Environment

1. **SSH into VM**:

   ```bash
   ssh -i ~/.ssh/azure_ml_key azureuser@<VM_PUBLIC_IP>
   ```

2. **Update Packages**:

   ```bash
   sudo apt update
   sudo apt install -y python3 python3-pip python3-venv
   ```

3. **Create Project Directory**:

   ```bash
   mkdir ~/deployed-ml-server
   cd ~/deployed-ml-server
   ```

### Step 3: Transfer Project Files

1. **Copy Files**:

   - From local machine, use `scp`:

     ```bash
     scp -i ~/.ssh/azure_ml_key -r /path/to/deployed-ml-server azureuser@<VM_PUBLIC_IP>:~/deployed-ml-server
     ```

   - Or use Git:

     ```bash
     git clone https://github.com/kershrita/deployed-ml-server.git
     ```

2. **Verify Files**:

   ```bash
   ls ~/deployed-ml-server
   ```

   - Expected: `app.py data models requirements.txt src train_model.py .env`

### Step 4: Set Up Virtual Environment

1. **Create and Activate venv**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   - Includes `flask`, `pandas`, `numpy`, `scikit-learn`, `python-dotenv`, `gunicorn`.

### Step 5: Configure .env

- Ensure `.env` exists in `~/deployed-ml-server`:

  ```
  API_KEY=auth-key
  ```

- Verify:

  ```bash
  cat .env
  ```

### Step 6: Train Model (If Needed)

- If `models/classification_model.pkl` is missing or incompatible:

  ```bash
  cd ~/deployed-ml-server
  python3 train_model.py
  ```

- Creates `models/classification_model.pkl` using `data/data.csv`.

### Step 7: Install and Configure Gunicorn

- From `~/deployed-ml-server`:

  ```bash
  source venv/bin/activate
  gunicorn --bind 0.0.0.0:5000 app:app
  ```

- This runs the Flask app (`app.py`, `app` is the Flask instance).

- Stop with `Ctrl+C`.

### Step 8: Create Systemd Service for Gunicorn

1. **Create Service File**:

   ```bash
   sudo nano /etc/systemd/system/ml-server.service
   ```

   - Add:

     ```
     [Unit]
     Description=Gunicorn instance for ML Server
     After=network.target
     
     [Service]
     User=azureuser
     Group=www-data
     WorkingDirectory=/home/azureuser/deployed-ml-server
     Environment="PATH=/home/azureuser/deployed-ml-server/venv/bin"
     Environment="PYTHONPATH=/home/azureuser/deployed-ml-server"
     ExecStart=/home/azureuser/deployed-ml-server/venv/bin/gunicorn --workers 3 --bind 0.0.0.0:5000 app:app
     
     [Install]
     WantedBy=multi-user.target
     ```

   - Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

2. **Enable and Start Service**:

   ```bash
   sudo systemctl start ml-server
   sudo systemctl enable ml-server
   ```

3. **Check Status**:

   ```bash
   sudo systemctl status ml-server
   ```

   - Should show `active (running)`.

4. **View Logs**:

   ```bash
   journalctl -u ml-server.service -b
   ```

### Step 9: Test in Production with Postman

1. **Configure Postman**:

   - **Method**: POST

   - **URL**: `http://<VM_PUBLIC_IP>:5000/predict`

   - **Headers**:

     - `Content-Type: application/json`
     - `X-API-KEY: auth-key`

   - **Body** (raw, JSON):

     ```json
     {
         "age": 30,
         "gender": "Male",
         "income": 50000,
         "days_on_platform": 20,
         "city": "London"
     }
     ```

2. **Send Request**:

   - Click **Send**.

   - Expected response (Status: 200):

     ```json
     {"prediction": 1}
     ```

3. **Error Cases**:

   - **Missing Field**:

     ```json
     {
         "age": 30,
         "gender": "Male",
         "income": 50000,
         "days_on_platform": 20
     }
     ```

     - Expected: `{"error": "Missing required fields"}` (Status: 400)

   - **Wrong API Key**:

     - Use `X-API-KEY: wrongkey`.
     - Expected: `{"error": "Invalid API key"}` (Status: 401)

### Step 10: Alternative Testing Methods

1. **Using curl**:

   ```bash
   curl -X POST http://<VM_PUBLIC_IP>:5000/predict \
   -H "Content-Type: application/json" \
   -H "X-API-KEY: auth-key" \
   -d '{"age": 30, "gender": "Male", "income": 50000, "days_on_platform": 20, "city": "London"}'
   ```

   - Expected: `{"prediction":1}`

2. **Using Python requests**:

   - Save as `test_api.py` locally:

     ```python
     import requests
     
     url = "http://<VM_PUBLIC_IP>:5000/predict"
     headers = {
         "Content-Type": "application/json",
         "X-API-KEY": "auth-key"
     }
     data = {
         "age": 30,
         "gender": "Male",
         "income": 50000,
         "days_on_platform": 20,
         "city": "London"
     }
     
     response = requests.post(url, json=data, headers=headers)
     print(response.json())
     ```

   - Run:

     ```bash
     python test_api.py
     ```

   - Expected: `{'prediction': 1}`

### API Documentation

- **Endpoint**: `/predict`

- **Method**: POST

- **Headers**:

  - `Content-Type: application/json`
  - `X-API-KEY: auth-key`

- **Body**:

  ```json
  {
      "age": float,
      "gender": "Male" or "Female",
      "income": float,
      "days_on_platform": float,
      "city": string
  }
  ```

- **Response**:

  - Success (200):

    ```json
    {"prediction": int}
    ```

  - Error (400, 401, 500):

    ```json
    {"error": "error message"}
    ```

## Troubleshooting

- **Service Not Running**:
  - Check status: `sudo systemctl status ml-server`
  - View logs: `journalctl -u ml-server.service -b`
  - Ensure `gunicorn` is installed: `pip install gunicorn`
- **Port Not Accessible**:
  - Verify port 5000: `sudo netstat -tuln | grep 5000`
  - Check Azure networking for inbound TCP 5000.
- **Model Not Found**:
  - Ensure `models/classification_model.pkl` exists.
  - Run `python3 train_model.py` if missing.
- **ModuleNotFoundError**:
  - Ensure `src/__init__.py` exists.
- **API Key Error**:
  - Check `.env` and match `X-API-KEY` in Postman.

## Security Notes

- **HTTPS**: The server runs on HTTP. For production, consider Nginx with SSL (not implemented here).
- **Firewall**: Restrict port 5000 to specific IPs in Azure’s network security group.
- **API Key**: Store securely and rotate periodically.
- **Backups**: Save `classification_model.pkl` and `data.csv` locally or in a repository.

## Clean Up

- **Stop VM**: In Azure Portal, stop `ml-server-vm` to save costs.
- **Delete Resources**: Delete `ml-server-rg` resource group to remove all resources.
