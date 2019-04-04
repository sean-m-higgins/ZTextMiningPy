

class ZettelPreProcessor: 
	def _init_(self, zettels):
		self.zettels = zettels

	def simple_tokenizer(zettel):
		tokens = zettel.split("(?U)[^\\p{Alpha}0-9']+").lower()
		


