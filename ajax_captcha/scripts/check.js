jQuery(function() {
jQuery("#form").validate({
rules: { 
captcha:	  { 
       		  required: true,
			 remote: {
        				url: "check.php",
        				type: "post"
						}
    
   				}, 
 		}, 
 
messages:	 { 
captcha: 		{ 
  					required: "Enter the characters as seen on the image above (case sensitive):",
 					remote: jQuery.format("Correct captcha is required.") 
  
  
				 }, 
			 },    			

 
submitHandler: function() {
			alert('valid string');
		}
	});
	});
