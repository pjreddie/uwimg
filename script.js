document.getElementById('imageInput').addEventListener('change', function(event) {
  const file = event.target.files[0];
  const image1 = document.getElementById('image1');

  const formData = new FormData();
  formData.append('file', file);
  
  //saturated
  fetch('http://127.0.0.1:5000/process_image1', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    if (response.ok) {
      return response.blob();
    } else {
      throw new Error('Network response was not ok.');
    }
  })
  .then(blob => {
    image1.src = URL.createObjectURL(blob);;
  })
  .catch(error => {
    console.error('There was an error processing the image:', error);
  });

  //Black and white
  fetch('http://127.0.0.1:5000/process_image2', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    if (response.ok) {
      return response.blob();
    } else {
      throw new Error('Network response was not ok.');
    }
  })
  .then(blob => {
    image2.src = URL.createObjectURL(blob);;
  })
  .catch(error => {
    console.error('There was an error processing the image:', error);
  });

});