<?php require('./config.php');?>
<html>
<head>
<script src="./scripts/jquery.js" type="text/javascript"></script>
<script src="./scripts/jquery.form.js" type="text/javascript"></script>
<script src="./scripts/jquery.validate.js" type="text/javascript"></script>
<script src="./scripts/check.js" type="text/javascript"></script>
<style type="text/css" media="all ">
div.error {
display:none;
padding:0px 30px 0px;
margin-left:8px;
color:red;

}
form label.error {
	display: inline-block;
	/*line-height: 1.8;*/
	vertical-align: top;
	cursor: hand;
}
</style>
</head>
<body>
<div id="result" style="padding:8px; font-weight:bold;"></div>
<div class="error"><span></span></div>
<form name="form" id="form" action= "" method="post"> 
<table border="0" cellspacing="0" cellpadding="0" class="form">
<tr>
<td width="132" class="validimage"><?php echo $capimage; ?></td>
<td width="208"><input type="text" name="captcha" id ="captcha" class="captcha"/></td>
<td   class="buttonsubmit"><input  type="submit" value ="check" /></td>
</tr>
</table>
</form>
</body>
</html>
