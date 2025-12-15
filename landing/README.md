# HybridRAG Landing Page

A modern, animated landing page for HybridRAG inspired by [slush.app](https://slush.app/).

## Features

- Dark theme with vibrant accent colors
- Smooth scroll using [Lenis](https://lenis.studiofreight.com/)
- Scroll-triggered reveal animations
- Responsive design (mobile-first)
- 3D hover effects
- Glass-morphism cards
- Optimized performance

## Quick Start

### Local Development

Simply open `index.html` in your browser:

```bash
cd landing
open index.html  # macOS
# or
start index.html  # Windows
# or
xdg-open index.html  # Linux
```

Or use a local server for best results:

```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx serve

# Using PHP
php -S localhost:8000
```

Then visit `http://localhost:8000`

### Deploy to Replit

1. Create a new Replit project
2. Choose "HTML, CSS, JS" template
3. Upload all files from the `landing/` directory
4. The site will automatically deploy

### Deploy to GitHub Pages

1. Create a repository
2. Push the `landing/` directory contents
3. Go to Settings > Pages
4. Select "Deploy from a branch"
5. Choose your branch and `/` (root) folder

### Deploy to Vercel/Netlify

1. Push to GitHub
2. Connect your repository to Vercel/Netlify
3. Set the root directory to `landing/`
4. Deploy automatically

## File Structure

```
landing/
├── index.html          # Main HTML file
├── css/
│   ├── style.css       # Core styles and design system
│   ├── animations.css  # Keyframes and animation classes
│   └── responsive.css  # Mobile breakpoints
├── js/
│   ├── main.js         # Core interactions
│   └── animations.js   # Scroll animations
├── assets/
│   ├── favicon.svg     # Site favicon
│   └── icons/          # Feature icons
└── README.md           # This file
```

## Customization

### Colors

Edit CSS variables in `css/style.css`:

```css
:root {
    --primary-violet: #8B5CF6;
    --primary-blue: #3B82F6;
    --primary-green: #10B981;
    --primary-amber: #F59E0B;
    --bg-dark: #0A0A0B;
    /* ... */
}
```

### Content

Edit the HTML in `index.html`:
- Hero section tagline
- Feature descriptions
- Code examples
- Footer links

### Animations

Adjust animation timing in `css/animations.css` and `js/animations.js`.

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13.1+
- Edge 80+

## Performance

- No build step required
- Minimal external dependencies (only Lenis for smooth scroll)
- CSS animations (hardware accelerated)
- Lazy loading support
- Reduced motion support

## License

Apache 2.0 - Same as HybridRAG
