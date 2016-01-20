<?php
session_start();

if(($_REQUEST['captcha']) == $_SESSION['key'])
	{
	echo "true";
	}
	else
	{
	echo "false";
	}
	?>