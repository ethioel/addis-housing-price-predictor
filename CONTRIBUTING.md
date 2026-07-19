# 🤝 Contributing to Addis Ababa Housing Price Predictor

First off, thank you for considering contributing! 🎉 We welcome contributions from everyone, whether it's fixing a bug, improving documentation, or suggesting a new feature.

## 🐛 Found a Bug?

1. Check if it's already reported in [Issues](https://github.com/ethioel/addis-housing-price-predictor/issues)
2. If not, open a new issue with:
   - What happened
   - Steps to reproduce
   - Expected behavior
   - Your environment (OS, Python version)

## 💡 Have an Idea?

Open a [Feature Request](https://github.com/ethioel/addis-housing-price-predictor/issues/new) with:
- What you want to add
- Why it's useful
- How you think it could work

## 💻 Want to Contribute Code?

### 1. Fork & Clone

```bash
# Fork on GitHub first, then:
git clone https://github.com/yourusername/addis-housing-price-predictor.git
cd addis-housing-price-predictor
```
### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---


### 4. Make Your Changes

- Follow the existing code style
- Add tests if you're adding features
- Update documentation if needed
  
### 5. Run Checks

```bash
# Format code
black src/ tests/
isort src/ tests/

# Run tests
pytest tests/ -v

# Check linting
flake8 src/ tests/
```

---

### 6. Commit & Push

```bash
git add .
git commit -m "feat: add your feature description"
git push origin your-branch-name

```
---

### 7. Open a Pull Request

Go to the repository on GitHub and click "New Pull Request".

## 📝 Commit Message Guidelines

```text
feat: add new feature
fix: fix a bug
docs: update documentation
test: add or update tests
refactor: refactor code
style: code style changes
chore: maintenance tasks
```

---

## ✅ Before Submitting

- [ ] Code is formatted (`black` + `isort`)
- [ ] Tests pass (`pytest`)
- [ ] No linting errors (`flake8`)
- [ ] Documentation is updated
- [ ] Commit messages are clear

## 🤔 Questions?

Open a [Discussion](https://github.com/ethioel/addis-housing-price-predictor/discussions) 
or email: samkahsay.dev@gmail.com



