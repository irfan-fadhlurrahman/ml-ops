import requests

# notes:
# for production environment, use gunicorn library to activate
# the server.
# gunicorn --bind 0.0.0.0:9696 predict:app

customer = {
    'customer_id': '9412-GHEEC',
    'gender': 'Male',
    'senior_citizen': 0,
    'partner': 'No',
    'dependents': 'No',
    'tenure': 40,
    'phone_service': 'Yes',
    'multiple_lines': 'Yes',
    'internet_service': 'Fiber optic',
    'online_security': 'No',
    'online_backup': 'No',
    'device_protection': 'Yes',
    'tech_support': 'Yes',
    'streaming_tv': 'Yes',
    'streaming_movies': 'Yes',
    'contract': 'Month-to-month',
    'paperless_billing': 'No',
    'payment_method': 'Bank transfer (automatic)',
    'monthly_charges': 104.8,
    'total_charges': 4131.95
}

url = "http://localhost:9696/predict"
print(requests.post(url, json=customer).json())