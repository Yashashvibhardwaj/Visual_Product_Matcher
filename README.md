# Visual Product Matcher

A web application that helps users find visually similar products based on an uploaded image.  
This project was built as part of a technical assessment and focuses on creating a simple, intuitive, and responsive product-matching experience.

---

## Features

- **Image Upload Options**

  - Upload a local file
  - Paste an image URL

- **Search Interface**

  - Preview uploaded image
  - View a list of visually similar products
  - Filter results by similarity score

- **Product Database**

  - Contains 50+ sample products with images
  - Each product includes basic metadata: name, category, price, etc.

- **User Experience Enhancements**
  - Loading states for smooth interaction
  - Basic error handling for invalid uploads
  - Mobile-responsive design

---

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript (responsive UI)
- **Backend:** [Your backend framework choice, e.g., Node.js / Flask / Django]
- **Database:** JSON / SQLite / MongoDB (for product metadata)
- **Image Similarity:** [Mention if you used CLIP, TensorFlow, or any API like Pinecone, Weaviate, etc.]
- **Hosting:** [e.g., Vercel, Netlify, Render, or Heroku free tier]

---

## Project Structure

```
├── backend/          # API logic, image processing, similarity search
├── index.html         # Static files (HTML, CSS, JS)
├── README.md         # Documentation
```

---

## How It Works

1. User uploads an image or provides a URL.
2. Backend extracts image embeddings using an ML model or API.
3. Similarity search runs against the product database.
4. Results are displayed in a clean, filterable UI.

---

## Documentation

**Approach (Summary):**  
I focused on building a lightweight yet functional application that demonstrates real-world product-matching workflows. The system extracts features from uploaded images, compares them with pre-indexed product embeddings, and returns the closest matches. The UI is designed to be clean, responsive, and user-friendly, with loading states and error handling for better usability.

---

## Running Locally

```
# Clone repo
git clone https://github.com/Yashashvibhardwaj/visual-product-matcher.git

# Navigate
cd visual-product-matcher

# Install dependencies
npm install    # or pip install -r requirements.txt

# Start the backend
npm run start  # or python app.py

# Open frontend
http://localhost:3000
```

---

## Deliverables

- Working application (deployed)
- GitHub repository with clean, documented code
- Short write-up (approach summary)

---

## Author

Built by Yashashvi Bhardwaj
