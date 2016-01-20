#!/usr/bin/perl -w

my @transitionMatrix;

sub analyzeWord {
	my @chars = split(//, $_[0]);
	my $lastchar = -1;

	foreach my $char (@chars) {
		my $v = ord($char);

		if ($v >= 96) {
			$v -= 32;
		}

		$v -= 65;
		if (($v >= 0) && ($v < 26)) {
			if ($lastchar >= 0) {
				# Put into matrix
				my $place = $lastchar * 26 + $v;
				$transitionMatrix[$place]++;
			}
			$lastchar = $v;
		}
	}
}

sub analyzeLine {
	my @words = split(/ /, $_[0]);

	foreach my $word (@words) {
		analyzeWord($word);
	}
}

for(my $y = 0; $y < 26; $y++) {
	for(my $x = 0; $x < 26; $x++) {
		$transitionMatrix[$y * 26 + $x] = 0;
	}
}
open(FP, "8dgry10.txt");

while ($line=<FP>) {
	analyzeLine($line);
}

close(FP);

# Normalize
for(my $y = 0; $y < 26; $y++) {
	my $sum = 0;
	for(my $x = 0; $x < 26; $x++) {
		$sum +=	$transitionMatrix[$y * 26 + $x];
	}
	for(my $x = 0; $x < 26; $x++) {
		$transitionMatrix[$y * 26 + $x] /= $sum;
	}
	$sum = 0;
	for(my $x = 0; $x < 26; $x++) {
		$sum += $transitionMatrix[$y * 26 + $x];
		$transitionMatrix[$y * 26 + $x] = $sum;
	}
}

# Print
for(my $y = 0; $y < 26; $y++) {
	for(my $x = 0; $x < 26; $x++) {
		$v = $transitionMatrix[$y * 26 + $x];
		printf("%0.4f, ", $v);
	}
	print "\n";
}
