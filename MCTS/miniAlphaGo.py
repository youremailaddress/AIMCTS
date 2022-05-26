import random, sys, pygame, time
from pygame.locals import *
from MCTS import *

FRAMERATE = 10
WIDTH = 640
HEIGHT = 480
SPACESIZE = 50
ANIMATIONSPEED = 25

XMARGIN = int((WIDTH - (8 * SPACESIZE)) / 2)
YMARGIN = int((HEIGHT - (8 * SPACESIZE)) / 2)

WHITE      = (255, 255, 255)
BLACK      = (  0,   0,   0)
GREEN      = (  0, 155,   0)
BRIGHTBLUE = (  0,  50, 255)

TEXTBGCOLOR1 = BRIGHTBLUE
TEXTBGCOLOR2 = GREEN
GRIDLINECOLOR = BLACK
TEXTCOLOR = WHITE


def main():
    global MAINCLOCK, DISPLAYSURF, FONT, BIGFONT, BGIMAGE
    pygame.init()
    MAINCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Mini AlphaGo')
    FONT = pygame.font.Font('freesansbold.ttf', 16)
    BIGFONT = pygame.font.Font('freesansbold.ttf', 32)
    boardImage = pygame.image.load('flippyboard.png')
    boardImage = pygame.transform.smoothscale(boardImage, (8 * SPACESIZE, 8 * SPACESIZE))
    boardImageRect = boardImage.get_rect()
    boardImageRect.topleft = (XMARGIN, YMARGIN)
    BGIMAGE = pygame.image.load('flippybackground.png')
    BGIMAGE = pygame.transform.smoothscale(BGIMAGE, (WIDTH, HEIGHT))
    BGIMAGE.blit(boardImage, boardImageRect)
    while True:
        if runGame() == False:
            break

def runGame():
    mainBoard = Board()
    turn = random.choice(['computer', 'player'])
    drawBoard(mainBoard)
    playerTile, computerTile = setPlayerColor()
    newGameSurf = FONT.render('New Game', True, TEXTCOLOR, TEXTBGCOLOR2)
    newGameRect = newGameSurf.get_rect()
    newGameRect.topright = (WIDTH - 8, 10)
    lastcomptime = 0
    while True:
        if turn == 'player':
            if mainBoard.get_legal_actions(playerTile) == []:
                break
            movexy = None
            while movexy == None:
                drawBoard(mainBoard)
                drawInfo(mainBoard, playerTile, computerTile, turn,lastcomptime)
                DISPLAYSURF.blit(newGameSurf, newGameRect)
                MAINCLOCK.tick(FRAMERATE)
                pygame.display.update()
                checkForQuit()
                for event in pygame.event.get():
                    if event.type == MOUSEBUTTONUP:
                        mousex, mousey = event.pos
                        if newGameRect.collidepoint( (mousex, mousey) ):
                            return True
                        movexy = getSpaceClicked(mousex, mousey)
                        if movexy != None and not mainBoard._can_filp(playerTile,movexy):
                            movexy = None
            makeMove(mainBoard, playerTile, movexy[0], movexy[1], True)
            
            if mainBoard.get_legal_actions(computerTile) != []:
                turn = 'computer'
        else:
            if mainBoard.get_legal_actions(computerTile) == []:
                break
            drawBoard(mainBoard)
            drawInfo(mainBoard, playerTile, computerTile, turn,lastcomptime)
            DISPLAYSURF.blit(newGameSurf, newGameRect)
            MAINCLOCK.tick(FRAMERATE)
            pygame.display.update()
            a = time.time()
            tree = MTCSTree()
            x,y = tree.decision(mainBoard, computerTile, 500)
            b = time.time()
            lastcomptime = b-a
            makeMove(mainBoard, computerTile, x, y, True)
            if mainBoard.get_legal_actions(playerTile) != []:
                turn = 'player'

    # Display the final score.
    drawBoard(mainBoard)
    flag,diff = mainBoard.get_winner()
    if flag == playerTile:
    # Determine the text of the message to display.
        text = 'You beat the computer by %s points! Congratulations!' % \
               (diff)
    elif flag == computerTile:
        text = 'You lost. The computer beat you by %s points.' % \
               (diff)
    else:
        text = 'The game was a tie!'

    textSurf = FONT.render(text, True, TEXTCOLOR, TEXTBGCOLOR1)
    textRect = textSurf.get_rect()
    textRect.center = (int(WIDTH / 2), int(HEIGHT / 2))
    DISPLAYSURF.blit(textSurf, textRect)
    text2Surf = BIGFONT.render('Play again?', True, TEXTCOLOR, TEXTBGCOLOR1)
    text2Rect = text2Surf.get_rect()
    text2Rect.center = (int(WIDTH / 2), int(HEIGHT / 2) + 50)
    yesSurf = BIGFONT.render('Yes', True, TEXTCOLOR, TEXTBGCOLOR1)
    yesRect = yesSurf.get_rect()
    yesRect.center = (int(WIDTH / 2) - 60, int(HEIGHT / 2) + 90)
    noSurf = BIGFONT.render('No', True, TEXTCOLOR, TEXTBGCOLOR1)
    noRect = noSurf.get_rect()
    noRect.center = (int(WIDTH / 2) + 60, int(HEIGHT / 2) + 90)

    while True:
        checkForQuit()
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONUP:
                mousex, mousey = event.pos
                if yesRect.collidepoint( (mousex, mousey) ):
                    return True
                elif noRect.collidepoint( (mousex, mousey) ):
                    return False
        DISPLAYSURF.blit(textSurf, textRect)
        DISPLAYSURF.blit(text2Surf, text2Rect)
        DISPLAYSURF.blit(yesSurf, yesRect)
        DISPLAYSURF.blit(noSurf, noRect)
        pygame.display.update()
        MAINCLOCK.tick(FRAMERATE)


def translateBoardToPixelCoord(x, y):
    return XMARGIN + x * SPACESIZE + int(SPACESIZE / 2), YMARGIN + y * SPACESIZE + int(SPACESIZE / 2)

def animateTileChange(tilesToFlip, tileColor, additionalTile):
    if tileColor == 1:
        additionalTileColor = WHITE
    else:
        additionalTileColor = BLACK
    additionalTileX, additionalTileY = translateBoardToPixelCoord(additionalTile[0], additionalTile[1])
    pygame.draw.circle(DISPLAYSURF, additionalTileColor, (additionalTileX, additionalTileY), int(SPACESIZE / 2) - 4)
    pygame.display.update()

    for rgbValues in range(0, 255, int(ANIMATIONSPEED * 2.55)):
        if rgbValues > 255:
            rgbValues = 255
        elif rgbValues < 0:
            rgbValues = 0

        if tileColor == 1:
            color = tuple([rgbValues] * 3)
        elif tileColor == -1:
            color = tuple([255 - rgbValues] * 3)

        for x, y in tilesToFlip:
            centerx, centery = translateBoardToPixelCoord(x, y)
            pygame.draw.circle(DISPLAYSURF, color, (centerx, centery), int(SPACESIZE / 2) - 4)
        pygame.display.update()
        MAINCLOCK.tick(FRAMERATE)
        checkForQuit()

def drawBoard(board):
    board = board._board
    DISPLAYSURF.blit(BGIMAGE, BGIMAGE.get_rect())
    for x in range(9):
        startx = (x * SPACESIZE) + XMARGIN
        starty = YMARGIN
        endx = (x * SPACESIZE) + XMARGIN
        endy = YMARGIN + (8 * SPACESIZE)
        pygame.draw.line(DISPLAYSURF, GRIDLINECOLOR, (startx, starty), (endx, endy))
    for y in range(9):
        startx = XMARGIN
        starty = (y * SPACESIZE) + YMARGIN
        endx = XMARGIN + (8 * SPACESIZE)
        endy = (y * SPACESIZE) + YMARGIN
        pygame.draw.line(DISPLAYSURF, GRIDLINECOLOR, (startx, starty), (endx, endy))

    for x in range(8):
        for y in range(8):
            centerx, centery = translateBoardToPixelCoord(x, y)
            if board[x][y] == 1 or board[x][y] == -1:
                if board[x][y] == 1:
                    tileColor = WHITE
                else:
                    tileColor = BLACK
                pygame.draw.circle(DISPLAYSURF, tileColor, (centerx, centery), int(SPACESIZE / 2) - 4)

def getSpaceClicked(mousex, mousey):
    for x in range(8):
        for y in range(8):
            if mousex > x * SPACESIZE + XMARGIN and \
               mousex < (x + 1) * SPACESIZE + XMARGIN and \
               mousey > y * SPACESIZE + YMARGIN and \
               mousey < (y + 1) * SPACESIZE + YMARGIN:
                return (x, y)
    return None

def drawInfo(board, playerTile, computerTile, turn,tms):
    flag,diff = board.get_winner()
    if flag == playerTile:
        scoreSurf = FONT.render("%s's Turn,Player is %s point ahead of Computer,LastComputeTime=%5f" %(turn.title(),diff,tms), True, TEXTCOLOR)
    elif flag == computerTile:
        scoreSurf = FONT.render("%s's Turn,Computer is %s point ahead of Player,LastComputeTime=%5f" %(turn.title(),diff,tms), True, TEXTCOLOR)
    elif flag == 0:
        scoreSurf = FONT.render("%s's Turn,Computer draw with Player,LastComputeTime=%5f" %(turn.title(),tms), True, TEXTCOLOR)
    scoreRect = scoreSurf.get_rect()
    scoreRect.bottomleft = (10, HEIGHT - 5)
    DISPLAYSURF.blit(scoreSurf, scoreRect)

def setPlayerColor():
    textSurf = FONT.render('Do you want to be white or black?', True, TEXTCOLOR, TEXTBGCOLOR1)
    textRect = textSurf.get_rect()
    textRect.center = (int(WIDTH / 2), int(HEIGHT / 2))

    xSurf = BIGFONT.render('White', True, TEXTCOLOR, TEXTBGCOLOR1)
    xRect = xSurf.get_rect()
    xRect.center = (int(WIDTH / 2) - 60, int(HEIGHT / 2) + 40)

    oSurf = BIGFONT.render('Black', True, TEXTCOLOR, TEXTBGCOLOR1)
    oRect = oSurf.get_rect()
    oRect.center = (int(WIDTH / 2) + 60, int(HEIGHT / 2) + 40)

    while True:
        checkForQuit()
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONUP:
                mousex, mousey = event.pos
                if xRect.collidepoint( (mousex, mousey) ):
                    return [1, -1]
                elif oRect.collidepoint( (mousex, mousey) ):
                    return [-1, 1]

        DISPLAYSURF.blit(textSurf, textRect)
        DISPLAYSURF.blit(xSurf, xRect)
        DISPLAYSURF.blit(oSurf, oRect)
        pygame.display.update()
        MAINCLOCK.tick(FRAMERATE)

def makeMove(board, tile, xstart, ystart, realMove=False):
    # Place the tile on the board at xstart, ystart, and flip tiles
    # Returns False if this is an invalid move, True if it is valid.
    tilesToFlip = board._can_filped((xstart,ystart), tile)
    if tilesToFlip == False:
        return False
    board._board[xstart][ystart] = tile
    if realMove:
        animateTileChange(tilesToFlip, tile, (xstart, ystart))
    for x, y in tilesToFlip:
        board._board[x][y] = tile
    return True
    

    board[xstart][ystart] = tile

    if realMove:
        animateTileChange(tilesToFlip, tile, (xstart, ystart))

    for x, y in tilesToFlip:
        board[x][y] = tile
    return True

def checkForQuit():
    for event in pygame.event.get((QUIT, KEYUP)): # event handling loop
        if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()

if __name__ == '__main__':
    main()
