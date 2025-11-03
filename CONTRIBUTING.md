# Contributing Guide

Thank you for considering contributing to the Enterprise RAG Q&A System! This guide will help you get started.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/repo-name.git
   cd repo-name
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment**
   ```bash
   cp .env.example .env
   # Add your GOOGLE_API_KEY
   ```

## Making Changes

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow existing code style
   - Add tests for new features
   - Update documentation

3. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

4. **Test Locally**
   ```bash
   streamlit run app.py
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for functions and classes
- Keep functions focused and small

## Testing

- Write unit tests for new functions
- Add integration tests for workflows
- Ensure all tests pass before submitting PR

## Commit Messages

Follow conventional commit format:

```
type(scope): subject

body

footer
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test changes
- `refactor`: Code refactoring
- `chore`: Build/config changes

Example:
```
feat(workflow): add hybrid search support

Implement hybrid search combining semantic and keyword search
for improved retrieval accuracy.

Closes #123
```

## Pull Request Process

1. **Update Documentation**
   - Update README.md if needed
   - Add deployment notes to DEPLOYMENT.md
   - Update CHANGELOG.md (if exists)

2. **Create Pull Request**
   - Use clear, descriptive title
   - Reference related issues
   - Describe changes in detail
   - Add screenshots for UI changes

3. **Code Review**
   - Address review comments
   - Keep discussion focused
   - Update PR based on feedback

4. **Merge**
   - Squash commits if needed
   - Ensure CI passes
   - Wait for maintainer approval

## Areas for Contribution

### High Priority
- [ ] Multi-user authentication system
- [ ] Advanced retriever options (hybrid search, reranking)
- [ ] PDF document support
- [ ] Conversation memory persistence

### Medium Priority
- [ ] Analytics dashboard
- [ ] Custom model configuration UI
- [ ] Batch document ingestion
- [ ] Export/import knowledge base

### Nice to Have
- [ ] Dark mode theme
- [ ] Multiple language support
- [ ] Voice input/output
- [ ] Mobile-responsive design

## Questions?

Feel free to:
- Open an issue for bugs
- Start a discussion for feature ideas
- Ask questions in issues

Thank you for contributing!
