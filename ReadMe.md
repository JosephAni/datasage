# Inventory Optimization

A Flask-based web application for inventory management and optimization, featuring various tools for demand forecasting, safety stock calculation, and inventory analysis.

## Features

- **Data Management**
  - File upload and processing
  - Data cleaning and preprocessing
  - Data type conversion
  - Data visualization

- **Inventory Analysis**
  - Safety stock simulation
  - Economic Order Quantity (EOQ) calculation
  - Inventory turnover analysis
  - Cost of inventory analysis

- **Demand Forecasting**
  - Time series analysis
  - Seasonal decomposition
  - Multiple forecasting methods
  - Visualization of forecasts

- **Machine Learning**
  - Classification and regression models
  - Feature engineering
  - Model training and evaluation
  - Prediction capabilities

## Installation

1. Clone the repository:

```bash
git clone https://github.com/JosephAni/datasage.git
cd datasage
```

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  
```

> **Note:**
>
> - Always activate the virtual environment with `source venv/bin/activate` before running the app or installing packages.
> - Install all requirements inside the venv:
>
>   ```bash
>   pip install -r requirements.txt
>   ```
>
> - If you use an IDE, set the Python interpreter to `venv/bin/python` for this project.

---

### Optional: Use a run.sh Script for Convenience

You can create a `run.sh` script to automatically activate the venv and run the app:

```bash
#!/bin/bash
source venv/bin/activate
python app.py
```

Make it executable:

```bash
chmod +x run.sh
```

Then run your app with:

```bash
./run.sh
```

1. Create a `.env` file with the following variables:

```env
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///inventory.db
DEBUG=True
```

1. Run the application:

```bash
python app.py
```

The application will be available at `http://localhost:8080`

## Project Structure

```plaintext
inventory-optimization/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── .gitignore            # Git ignore file
├── static/               # Static files (CSS, JS, images)
│   ├── css/
│   └── js/
└── templates/            # HTML templates
```

## Dependencies

- Flask 3.0.2
- Flask-SQLAlchemy 3.1.1
- pandas 2.2.1
- numpy 1.26.4
- scikit-learn 1.4.0
- matplotlib 3.8.2
- plotly 5.18.0
- And other dependencies listed in requirements.txt

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Joseph Ani - [GitHub](https://github.com/JosephAni)

Project Link: [https://github.com/JosephAni/datasage](https://github.com/JosephAni/datasage)

## To activate platform run  nix-shell dev.nix
