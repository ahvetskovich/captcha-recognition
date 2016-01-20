<?php

error_reporting(E_ALL); 

require('./class/captcha.php');

$width = '100';
$height = '30';
$characters = '6';
$imageurl = 'captcha/';
$imagepath = './captcha/';
$fontpath = './fonts/monofont.ttf';
$cap = new Captcha();
$capimage = $cap->create_captchaImage($fontpath,$imagepath,$imageurl,$width,$height,$characters);


?>