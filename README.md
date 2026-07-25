# 🏠 Addis Ababa Housing Price Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Tests](https://github.com/ethioel/addis-housing-price-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/ethioel/addis-housing-price-predictor/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://img.shields.io/badge/downloads-latest-brightgreen.svg)](https://github.com/ethioel/addis-housing-price-predictor/releases)

> **Synthetic housing price dataset generation for Addis Ababa, Ethiopia**  
> Based on real Jiji.com property listings with realistic distributions

---

## 📊 Overview

This project generates **realistic synthetic housing data** for Addis Ababa based on manually extracted property listings from Jiji.com. Since Jiji.com employs anti-bot protection that prevents web scraping, we manually extracted distribution patterns and built a synthetic data generator that preserves statistical properties while allowing scalability.

### 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **15,000+ Records** | Large, realistic dataset for ML modeling |
| **10 Sub-cities** | Complete coverage of Addis Ababa |
| **5 Property Types** | House, Apartment, Condo, Villa, Studio |
| **15+ Features** | Comprehensive property attributes |
| **Realistic Distributions** | Based on actual Jiji.com listings |
| **Reproducible** | Seeded random generation |

### 💡 Use Cases

- 🏗️ **Real Estate Price Prediction** - Train ML models on realistic data
- 📈 **Market Analysis** - Analyze property trends in Addis Ababa
- 🎓 **Data Science Education** - Perfect for teaching and learning
- 🧪 **Model Benchmarking** - Test algorithms on Ethiopian housing data

---

## 🎯 Why Synthetic Data?

| Challenge | Solution |
|-----------|--------------|
| ❌ **Jiji.com can't be scraped** (anti-bot protection) | ✅ Manual extraction of real distribution patterns |
| ❌ **Limited real data availability** | ✅ Scalable synthetic generation (any sample size) |
| ❌ **Privacy & ethical concerns** | ✅ No real property information exposed |
| ❌ **Data imbalance issues** | ✅ Controlled, balanced distributions |
| ❌ **Costly data collection** | ✅ Free, instant dataset generation |

---

📚 Documentation
- **Data Dictionary** - Complete feature descriptions with examples
- **API Reference** - Function documentation
- **Contributing Guide** - How to contribute

🤝 Contributing

We welcome contributions from the community! Here's how you can help:

- **Report Bugs** - Open an issue with detailed description
- **Suggest Features** - Propose new features or improvements
- **Submit PRs** - Fix bugs or add features
- **Improve Documentation** - Help make docs better
- **Share Use Cases** - Tell us how you're using the project

See `CONTRIBUTING.md` for detailed guidelines.

📝 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

🙏 Acknowledgments
- **Jiji.com.et** - For providing real-world property listing patterns
- **Kaggle Housing Dataset** - Inspiration for feature design
- **Ethiopian Data Science Community** - Support and feedback

📧 Contact & Support
| Role | Info |
|------|------|
| **Author** | Samuel Kahsay |
| **Email** | samkahsay.dev@gmail.com |
| **GitHub** | [@ethioel](https://github.com/ethioel) |
| **X/Twitter** | [@ethioel](https://X.com/ethioel) |
| **Project Link** | [github.com/ethioel/addis-housing-price-predictor](https://github.com/ethioel/addis-housing-price-predictor) |

**Support**
- 📖 Docs: `docs/data_dictionary.md`
- 🐛 Bugs: [GitHub Issues](https://github.com/ethioel/addis-housing-price-predictor/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/ethioel/addis-housing-price-predictor/discussions)

⭐ Show Your Support
If you find this project useful...

- ⭐ Star this repository
- 🐦 Share on Twitter/LinkedIn
- 📖 Use in your projects
- 🤝 Contribute to development

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ethioel/addis-housing-price-predictor.git
cd addis-housing-price-predictor

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in development mode
pip install -e .

# 5. Verify installation
python -c "from src import data_generator; print('✅ Success!')"
```
---
Built with ❤️ for the Ethiopian data science community
