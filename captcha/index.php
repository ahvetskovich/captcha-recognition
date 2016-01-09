<?php
  //--Проверяем если капча была введена
  if ( isset($_POST['capcha']) )
  {
    $code = $_POST['capcha']; //Получаем переданную капчу
    session_start();

    if ( isset($_SESSION['capcha']) && strtoupper($_SESSION['capcha']) == strtoupper($code) ) //сравниваем введенную капчу с сохраненной в переменной в сессии
      echo 'Верно!';
    else
      echo 'Не верно!';
    //Удаляем капчу из сессии 
    unset($_SESSION['capcha']);
    exit();
  }
?>

<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
  <head>
    <title>Тест</title>
    <meta http-equiv="Content-Type" content="text/html; charset=windows-1251">
  </head>
  <body>
    <form method="post" action="<?php echo $_SERVER['PHP_SELF']; ?>">
      <img src="captcha.php" width="120" height="20"><br>
      <input type="text" name="capcha"><br>
      <input type="submit" value="Отправить">
    </form>
  </body> 
</html>

