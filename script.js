document.getElementById('imageInput').addEventListener('change', function(event) {
  const file = event.target.files[0];
  const imagePreview = document.getElementById('imagePreview');
  imagePreview.src = URL.createObjectURL(file);
});