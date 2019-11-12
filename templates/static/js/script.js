$( document ).ready(function() {
    
	//	Board name that matches div
    var boardName = 'myBoard'
	
	//	Initialize chess js object
	var game = new Chess()
	
	//	Start chess as cleared
	game.clear()
	
    //  Hard code start configuration
	var positions = {}
    
    //  Initialize board configuration
	var boardConfig = {
		pieceTheme: '/static/{piece}.png',	//	location of piece images
		position:positions,					//	initial piece positions dictionary
		draggable: true,					//	pieces are draggable
		onSnapEnd: onSnapEnd,				//	Function to call after piece is snapped to a new position
		onDrop:onDrop,						//	Function to call piece is dropped (before snap)
		onChange:onChange,					//	Function to call when board position changes
		sparePieces: true					//	Draws spare black and white pieces on either side of the board
	}
    
    //  SVG for text below chess board
	var svgW = 400
	var svgH = 100
	var svgTextSize = 15
	var svgVerticalSpace = svgTextSize+3
	var svg = d3.select('#boardConfig')
		.append('svg')
		.attr("width", svgW)
		.attr("height", svgH)		
	
	//	Text object for listing positions
	var positionText = svg.append('text')
		.attr('x',0)
		.attr('y', 0)
		.attr('dominant-baseline', 'hanging')
		.attr('text-anchor', 'start')  
		.style('font-size', svgTextSize + 'px')
    
	//	Text object for listing values
    var valueText = svg.append('text')
		.attr('x',0)
		.attr('y', svgVerticalSpace)
		.attr('dominant-baseline', 'hanging')
		.attr('text-anchor', 'start')  
		.style('font-size', svgTextSize + 'px')
		
	var nextTurnText = svg.append('text')
		.attr('x',0)
		.attr('y', 2*svgVerticalSpace)
		.attr('dominant-baseline', 'hanging')
		.attr('text-anchor', 'start')  
		.style('font-size', svgTextSize + 'px')
		.text('Next Turn: ')
		
	var messageText = svg.append('text')
		.attr('x',0)
		.attr('y', 3*svgVerticalSpace)
		.attr('dominant-baseline', 'hanging')
		.attr('text-anchor', 'start')  
		.style('font-size', svgTextSize + 'px')
		.text('Message: ')
    
    //  Create chess board
	var board = Chessboard(boardName,boardConfig)
    
	//	Colorbar
    var colorBarWidth = 400
    var colorBarHeight = 20
	var numBars = 100
    var color = d3.interpolateViridis
    var colorBar = d3.select('#' + boardName)
        .append('svg')
        .attr('width', colorBarWidth)
        .attr('height',colorBarHeight)
    for (var b = 0; b < numBars; b++) {
        colorBar.append('rect')
        .attr('x', b * colorBarWidth/numBars)
        .attr('y', 0)
        .attr('width', colorBarWidth/numBars)
        .attr('height',20)
        .style('fill',color(b/numBars))
    }
	
	//	Clears custom color under a single square
	function clearSquare(square){
		$('#myBoard .square-' + square).css('background', '')
	}

    //  Function ran when piece moves
	function onSnapEnd(source=null, target=null, piece=null){	
		
		//	Get current board positions
		var boardPositions = board.position()
		
		//	Update positions text object
		positionText.text("Positions:\t" + JSON.stringify(boardPositions).replace(/\"/g,'').replace(/\{/g,'').replace(/\}/g,'').replace(/\,/g,', '))
		
		//	POST to python and receive values
		$.post( "/postmethod", boardPositions, function(err, req, resp){
			
			//	Receive piece values dictionary
            var pieceValues = resp.responseJSON
			
			//	Round values for display in valueText object
			var roundedValues = pieceValues
			for (i in roundedValues)
				roundedValues[i] = roundedValues[i].toFixed(2)
			valueText.text("Values:\t" + JSON.stringify(roundedValues).replace(/\"/g,'').replace(/\{/g,'').replace(/\}/g,'').replace(/\,/g,', '))
			
			//	Loop over piece values and draw colors in appropriate squares
            for (var s in pieceValues) {
				$('#myBoard .square-' + s).css('background', color(pieceValues[s]))
            }
		});
		
		//	If piece isn't added from spare pieces and source isn't null, clear source square
		//	source is only null in manual run of function below, to initialize the values
		if (source != null && source != 'spare') {
			clearSquare(source)
		}
	}
	
	//	Function to run on drop of piece (before onSnapEnd)
	function onDrop(source, target, piece, newPos, oldPos, orientation){
		
		//	If moved off board, remove from chessjs object, clear square color, return 'trash'
		if (target == 'offboard') {
			game.remove(source)
			clearSquare(source)
			messageText.text('Message: Piece manually removed, circumventing rules')
			return 'trash'
		}
		
		//	If piece is already there, snap back
		if (board.position()[target] == piece) {
			return 'snapback'
		}
		
		//	If piece coming from spares, add it to chessjs game
		if (source == 'spare') {
			tryPut = game.put({ type: piece[1].toLowerCase(), color: piece[0] }, target)
			
			//	If not allowed to put on chessjs board, snap back
			if (!tryPut) {
				return 'snapback'
			}
			messageText.text('Message: Piece manually added, circumventing rules')
		}
		
		//	If moving piece on the board, try making move in chessjs object
		else {
			//	Try moving piece, assuming queen promotion if necessary
			tryMove = game.move({to:target , from:source, promotion:'q'})
			
			//	If move not allowed, snap back
			if (!tryMove) {
				messageText.text('Message: Move not legal')
				return 'snapback'
			}
			messageText.text('Message: Legal move made')
		}
	}
	
	function onChange(){
		var nextMove = game.turn()
		var nextMoveText
		if (nextMove == 'b')
			nextMoveText = 'Black'
		else
			nextMoveText = 'White'
		nextTurnText.text('Next Turn: ' + nextMoveText)
	}
	
	// Run Function
    onSnapEnd()
  
});
